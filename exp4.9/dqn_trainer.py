"""
DQN Trainer Module for Exp4.9: Regime-Conditioned LESR

Modified from Exp4.7:
- extract_state now also computes regime_vector via regime_detector
- revise_state is called with (raw_state, regime_vector) 
- evaluate reports per-regime metrics
- DQN network architecture is UNCHANGED
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Optional, Callable, Dict, List, Tuple
import logging
import pickle
import copy

from regime_detector import detect_regime, classify_regime

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network for discrete action space (buy/sell/hold). UNCHANGED from 4.7."""

    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 256, device: str = None):
        super(DQN, self).__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()


class DQNTrainer:
    """
    DQN Trainer for financial trading with regime conditioning.

    Key change from 4.7: revise_state_func is called as f(raw_state, regime_vector)
    """

    def __init__(
        self,
        ticker: str,
        revise_state_func: Callable,
        intrinsic_reward_func: Callable,
        state_dim: int,
        intrinsic_weight: float = 0.02,
        device: str = None
    ):
        self.ticker = ticker
        self.revise_state = revise_state_func
        self.intrinsic_reward = intrinsic_reward_func
        self.intrinsic_weight = intrinsic_weight

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Networks — UNCHANGED architecture
        self.dqn = DQN(state_dim, device=self.device).to(self.device)
        self.target_dqn = DQN(state_dim, device=self.device).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()

        # Hyperparameters — UNCHANGED
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        # For feature analysis
        self.episode_states: List[np.ndarray] = []
        self.episode_rewards: List[float] = []
        self.episode_regimes: List[str] = []  # NEW: track regime labels

        # Cache
        self._state_cache = {}

    def calculate_reward(self, current_price: float, next_price: float) -> float:
        return (next_price - current_price) / current_price

    def extract_state(
        self,
        data_loader,
        date: str,
        window: int = 20
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract 20-day window of OHLCV data as raw state AND compute regime_vector.

        Returns:
            (raw_state, regime_vector) or (None, None) if insufficient data
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

    def _get_cached_state(self, data_loader, date):
        """Cache enhanced state computation."""
        date_str = str(date)
        if date_str not in self._state_cache:
            raw_state, regime_vector = self.extract_state(data_loader, date)
            if raw_state is not None:
                # KEY CHANGE: pass regime_vector to revise_state
                enhanced = self.revise_state(raw_state, regime_vector)
                # Ensure consistent dimension
                if enhanced.shape[0] != self.dqn.network[0].in_features:
                    # Pad or truncate to expected dim
                    expected = self.dqn.network[0].in_features
                    if enhanced.shape[0] < expected:
                        enhanced = np.concatenate([enhanced, np.zeros(expected - enhanced.shape[0])])
                    else:
                        enhanced = enhanced[:expected]
                self._state_cache[date_str] = (enhanced, classify_regime(regime_vector))
            else:
                self._state_cache[date_str] = None
        return self._state_cache[date_str]

    def train(
        self,
        train_data_loader,
        start_date: str,
        end_date: str,
        max_episodes: int = 100
    ) -> Dict:
        """Train DQN on training data."""
        dates = [
            d for d in train_data_loader.get_date_range()
            if start_date <= str(d) <= end_date
        ]

        logger.info(f"Training on {len(dates)} dates from {start_date} to {end_date}")
        print(f"Training on {len(dates)} dates from {start_date} to {end_date}")

        # Precompute all states
        logger.info("Precomputing features with regime detection...")
        print("Precomputing features with regime detection...")
        for date in dates:
            self._get_cached_state(train_data_loader, date)
        logger.info(f"Feature computation done, cached {len(self._state_cache)} states")
        print(f"Feature computation done, cached {len(self._state_cache)} states")

        for episode in range(max_episodes):
            episode_reward = 0
            epsilon = max(self.epsilon_end, self.epsilon * (self.epsilon_decay ** episode))

            for i, date in enumerate(dates):
                cached = self._get_cached_state(train_data_loader, date)
                if cached is None:
                    continue
                enhanced_state, regime_label = cached

                # Select action
                action = self.dqn.select_action(enhanced_state, epsilon)

                # Calculate reward
                current_price = train_data_loader.get_ticker_price_by_date(self.ticker, date)
                if i < len(dates) - 1:
                    next_date = dates[i + 1]
                    next_price = train_data_loader.get_ticker_price_by_date(self.ticker, next_date)
                    reward = self.calculate_reward(current_price, next_price)

                    # Add intrinsic reward
                    intrinsic_r = self.intrinsic_reward(enhanced_state)
                    total_reward = reward + self.intrinsic_weight * intrinsic_r

                    next_cached = self._get_cached_state(train_data_loader, next_date)
                    if next_cached is None:
                        next_enhanced = enhanced_state
                    else:
                        next_enhanced, _ = next_cached
                else:
                    next_enhanced = enhanced_state
                    total_reward = 0

                # Store experience
                self.buffer.push(enhanced_state, action, total_reward, next_enhanced, False)

                # Record for feature analysis (with regime label)
                self.episode_states.append(enhanced_state.copy())
                self.episode_rewards.append(total_reward)
                self.episode_regimes.append(regime_label)

                # Train network
                if len(self.buffer) > self.batch_size:
                    self._update_network()

                episode_reward += total_reward

            self._soft_update_target()

            print(f"Episode {episode}/{max_episodes}, Epsilon: {epsilon:.3f}, Buffer: {len(self.buffer)}", flush=True)
            logger.info(f"Episode {episode}/{max_episodes}, Epsilon: {epsilon:.3f}, Buffer: {len(self.buffer)}")

        return self._get_training_summary()

    def _update_network(self) -> None:
        """Update Q-network. UNCHANGED."""
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_dqn(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _soft_update_target(self) -> None:
        """Soft update target network. UNCHANGED."""
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _get_training_summary(self) -> Dict:
        return {
            'states': self.episode_states,
            'rewards': self.episode_rewards,
            'regimes': self.episode_regimes
        }

    def get_state_dict_portable(self) -> dict:
        return {k: v.cpu() for k, v in self.dqn.state_dict().items()}

    def load_state_dict_portable(self, state_dict: dict):
        self.dqn.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def evaluate(
        self,
        val_data_loader,
        start_date: str,
        end_date: str
    ) -> Dict:
        """Evaluate DQN on validation data. Returns per-regime metrics."""
        dates = [
            d for d in val_data_loader.get_date_range()
            if start_date <= str(d) <= end_date
        ]

        total_return = 0
        trades = []
        current_position = 0
        entry_price = 0
        daily_returns = []
        prev_price = None

        # Per-regime tracking
        regime_returns = {}
        regime_trades = {}

        for date in dates:
            raw_state, regime_vector = self.extract_state(val_data_loader, date)
            if raw_state is None:
                continue

            enhanced_state = self.revise_state(raw_state, regime_vector)
            # Ensure consistent dimension
            expected_dim = self.dqn.network[0].in_features
            if enhanced_state.shape[0] != expected_dim:
                if enhanced_state.shape[0] < expected_dim:
                    enhanced_state = np.concatenate([enhanced_state, np.zeros(expected_dim - enhanced_state.shape[0])])
                else:
                    enhanced_state = enhanced_state[:expected_dim]
            regime_label = classify_regime(regime_vector)

            action = self.dqn.select_action(enhanced_state, epsilon=0.0)
            current_price = val_data_loader.get_ticker_price_by_date(self.ticker, date)

            if action == 0:  # Buy
                if current_position == 0:
                    current_position = 1
                    entry_price = current_price
                    trades.append(('buy', date, current_price, regime_label))
                    regime_trades[regime_label] = regime_trades.get(regime_label, 0) + 1
            elif action == 1:  # Sell
                if current_position == 1:
                    current_position = 0
                    trades.append(('sell', date, current_price, regime_label))

            if prev_price is not None:
                daily_return = (current_price - prev_price) / prev_price if current_position == 1 else 0.0
                daily_returns.append(daily_return)
                if regime_label not in regime_returns:
                    regime_returns[regime_label] = []
                regime_returns[regime_label].append(daily_return)

            prev_price = current_price

        # Calculate overall metrics
        sharpe = self._calculate_sharpe(daily_returns)
        max_dd = self._calculate_max_drawdown(daily_returns)
        total_return_pct = sum(daily_returns) * 100

        # Calculate per-regime metrics
        regime_metrics = {}
        for regime, rets in regime_returns.items():
            regime_metrics[regime] = {
                'sharpe': self._calculate_sharpe(rets),
                'trades': regime_trades.get(regime, 0),
                'total_return': sum(rets) * 100,
                'days': len(rets)
            }

        return {
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_return': total_return_pct,
            'trades': trades,
            'regime_metrics': regime_metrics  # NEW
        }

    def _calculate_sharpe(self, daily_returns: List[float], risk_free_rate: float = 0.0) -> float:
        if len(daily_returns) < 2:
            return 0.0
        returns = np.array(daily_returns)
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        if std_return == 0:
            return 0.0
        return (mean_return - risk_free_rate) / std_return

    def _calculate_max_drawdown(self, daily_returns: List[float]) -> float:
        if len(daily_returns) < 2:
            return 0.0
        cumulative = np.cumsum(daily_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max)
        return abs(drawdowns.min()) * 100
