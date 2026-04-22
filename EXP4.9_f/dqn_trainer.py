"""
DQN Trainer for Exp4.9_f

Key changes from 4.7:
- extract_state also computes regime_vector (3d)
- Framework prepends raw+regime to LLM features (dimension guaranteed)
- regime_bonus: soft reward for risk-aware actions
- Tracks worst trades for COT feedback
- DQN network architecture UNCHANGED
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Optional, Callable, Dict, List, Tuple
import logging

from regime_detector import detect_regime

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """UNCHANGED from 4.7."""
    def __init__(self, state_dim, action_dim=3, hidden_dim=256, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(self.device)

    def forward(self, x):
        return self.network(x)

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.forward(s).argmax().item()


class DQNTrainer:
    """
    DQN Trainer with regime conditioning.
    
    Key design: LLM's revise_state returns ONLY new features.
    Framework prepends raw_state(120) + regime_vector(3) automatically.
    """

    def __init__(
        self,
        ticker: str,
        revise_state_func: Callable,
        intrinsic_reward_func: Callable,
        state_dim: int,
        intrinsic_weight: float = 0.02,
        regime_bonus_weight: float = 0.005,
        device: str = None
    ):
        self.ticker = ticker
        self.revise_state = revise_state_func    # returns only new features
        self.intrinsic_reward = intrinsic_reward_func
        self.intrinsic_weight = intrinsic_weight
        self.regime_bonus_weight = regime_bonus_weight

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.dqn = DQN(state_dim, device=self.device).to(self.device)
        self.target_dqn = DQN(state_dim, device=self.device).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()

        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        self.episode_states = []
        self.episode_rewards = []
        self.episode_regimes = []

        # Worst trade tracking for COT
        self.worst_trades = []

        self._state_cache = {}

    def extract_state(self, data_loader, date, window=20):
        """
        Extract raw state and regime vector.
        Returns (raw_state, regime_vector) or (None, None).
        """
        dates = data_loader.get_date_range()
        try:
            idx = dates.index(date)
        except ValueError:
            return None, None
        if idx < window - 1:
            return None, None

        window_dates = dates[idx - window + 1:idx + 1]
        state_120d = []
        for d in window_dates:
            daily_data = data_loader.get_data_by_date(d)
            if self.ticker in daily_data.get('price', {}):
                price_dict = daily_data['price'][self.ticker]
                if isinstance(price_dict, dict):
                    state_120d.extend([
                        price_dict.get('close', 0),
                        price_dict.get('open', 0),
                        price_dict.get('high', 0),
                        price_dict.get('low', 0),
                        price_dict.get('volume', 0),
                        price_dict.get('adjusted_close', price_dict.get('close', 0))
                    ])
                else:
                    state_120d.extend([price_dict] * 6)

        if len(state_120d) < 120:
            state_120d.extend([0] * (120 - len(state_120d)))

        raw_state = np.array(state_120d[:120])
        regime_vector = detect_regime(raw_state)
        return raw_state, regime_vector

    def _build_enhanced_state(self, raw_state, regime_vector):
        """
        Framework-layer state construction: [raw(120) + regime(3) + llm_features(N)]
        This guarantees dimension consistency — LLM cannot mess it up.
        """
        try:
            llm_features = self.revise_state(raw_state)
            if not isinstance(llm_features, np.ndarray):
                llm_features = np.atleast_1d(np.array(llm_features, dtype=float))
            if llm_features.ndim != 1:
                llm_features = llm_features.flatten()
        except Exception as e:
            logger.warning(f"revise_state failed: {e}, using zeros")
            llm_features = np.zeros(3)

        return np.concatenate([raw_state, regime_vector, llm_features])

    def compute_regime_bonus(self, regime_vector, action):
        """
        Framework-layer regime bonus: soft reward for risk-aware actions.
        Only 2 rules, conservative:
          1. High risk + SELL → positive (encourage stop-loss)
          2. Extreme risk + BUY → negative (discourage entry)
        """
        risk_level = regime_vector[2]

        if risk_level > 0.7 and action == 1:   # SELL during elevated risk
            return 5.0
        if risk_level > 0.85 and action == 0:   # BUY during extreme risk
            return -5.0
        return 0.0

    def _get_cached_state(self, data_loader, date):
        date_str = str(date)
        if date_str not in self._state_cache:
            raw_state, regime_vector = self.extract_state(data_loader, date)
            if raw_state is not None:
                enhanced = self._build_enhanced_state(raw_state, regime_vector)
                self._state_cache[date_str] = enhanced
            else:
                self._state_cache[date_str] = None
        return self._state_cache[date_str]

    def train(self, data_loader, start_date, end_date, max_episodes=100):
        """Train DQN."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))

        dates = [d for d in data_loader.get_date_range()
                 if start_date <= str(d) <= end_date]

        logger.info(f"Training on {len(dates)} dates")
        print(f"Training on {len(dates)} dates", flush=True)

        # Precompute states
        logger.info("Precomputing features...")
        print("Precomputing features...", flush=True)
        for date in dates:
            self._get_cached_state(data_loader, date)
        logger.info(f"Cached {len(self._state_cache)} states")
        print(f"Cached {len(self._state_cache)} states", flush=True)

        for episode in range(max_episodes):
            epsilon = max(self.epsilon_end, self.epsilon * (self.epsilon_decay ** episode))

            for i, date in enumerate(dates):
                enhanced = self._get_cached_state(data_loader, date)
                if enhanced is None:
                    continue

                # Regime vector is at [120:123]
                regime_vector = enhanced[120:123]

                action = self.dqn.select_action(enhanced, epsilon)

                # Extrinsic reward
                current_price = data_loader.get_ticker_price_by_date(self.ticker, date)
                if i < len(dates) - 1:
                    next_date = dates[i + 1]
                    next_price = data_loader.get_ticker_price_by_date(self.ticker, next_date)
                    extrinsic_r = (next_price - current_price) / current_price if current_price > 0 else 0

                    # Intrinsic reward (LLM-generated)
                    intrinsic_r = self.intrinsic_reward(enhanced)

                    # Regime bonus (framework-layer)
                    regime_r = self.compute_regime_bonus(regime_vector, action)

                    total_reward = (extrinsic_r
                                   + self.intrinsic_weight * intrinsic_r
                                   + self.regime_bonus_weight * regime_r)

                    next_enhanced = self._get_cached_state(data_loader, next_date)
                    if next_enhanced is None:
                        next_enhanced = enhanced
                else:
                    next_enhanced = enhanced
                    total_reward = 0

                self.buffer.push(enhanced, action, total_reward, next_enhanced, False)

                # Track for analysis
                self.episode_states.append(enhanced.copy())
                self.episode_rewards.append(total_reward)
                self.episode_regimes.append(regime_vector.copy())

                # Track worst trades
                if action == 0:  # BUY
                    trade_return = extrinsic_r if i < len(dates) - 1 else 0
                    if len(self.worst_trades) < 50:
                        self.worst_trades.append({
                            'day': i, 'action': 'BUY', 'return': trade_return,
                            'trend': float(regime_vector[0]),
                            'vol': float(regime_vector[1]),
                            'risk': float(regime_vector[2])
                        })
                    elif trade_return < self.worst_trades[-1]['return']:
                        self.worst_trades.append({
                            'day': i, 'action': 'BUY', 'return': trade_return,
                            'trend': float(regime_vector[0]),
                            'vol': float(regime_vector[1]),
                            'risk': float(regime_vector[2])
                        })
                    self.worst_trades.sort(key=lambda x: x['return'])
                    self.worst_trades = self.worst_trades[:50]

                if len(self.buffer) > self.batch_size:
                    self._update_network()

            self._soft_update_target()
            print(f"Episode {episode}/{max_episodes}, eps={epsilon:.3f}, buf={len(self.buffer)}", flush=True)

        return self._get_summary()

    def _update_network(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.dqn(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_dqn(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _soft_update_target(self):
        for tp, p in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _get_summary(self):
        return {
            'states': self.episode_states,
            'rewards': self.episode_rewards,
            'regimes': self.episode_regimes,
            'worst_trades': self.worst_trades
        }

    def evaluate(self, data_loader, start_date, end_date):
        """Evaluate on test data."""
        dates = [d for d in data_loader.get_date_range()
                 if start_date <= str(d) <= end_date]

        daily_returns = []
        prev_price = None
        current_position = 0

        for date in dates:
            raw_state, regime_vector = self.extract_state(data_loader, date)
            if raw_state is None:
                continue

            enhanced = self._build_enhanced_state(raw_state, regime_vector)
            action = self.dqn.select_action(enhanced, epsilon=0.0)
            current_price = data_loader.get_ticker_price_by_date(self.ticker, date)

            if action == 0:
                current_position = 1
            elif action == 1:
                current_position = 0

            if prev_price is not None:
                dr = (current_price - prev_price) / prev_price if current_position == 1 else 0.0
                daily_returns.append(dr)

            prev_price = current_price

        sharpe = self._sharpe(daily_returns)
        max_dd = self._max_dd(daily_returns)
        total_ret = sum(daily_returns) * 100

        return {
            'sharpe': sharpe, 'max_dd': max_dd, 'total_return': total_ret,
            'trades': [], 'regime_metrics': {}
        }

    def _sharpe(self, returns, rf=0.0):
        if len(returns) < 2: return 0.0
        r = np.array(returns)
        m, s = r.mean() * 252, r.std() * np.sqrt(252)
        return (m - rf) / s if s > 0 else 0.0

    def _max_dd(self, returns):
        if len(returns) < 2: return 0.0
        cum = np.cumsum(returns)
        peak = np.maximum.accumulate(cum)
        return abs((cum - peak).min()) * 100

    def get_state_dict_portable(self):
        return {k: v.cpu() for k, v in self.dqn.state_dict().items()}

    def load_state_dict_portable(self, sd):
        self.dqn.load_state_dict({k: v.to(self.device) for k, v in sd.items()})
        self.target_dqn.load_state_dict(self.dqn.state_dict())
