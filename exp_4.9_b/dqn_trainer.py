"""
DQN Trainer Module for Exp4.9_b Financial Trading Experiment

Changes from exp4.7:
- B1: Position-aware state (append position flag to state vector)
- B2: Transaction cost penalty in reward
- C1: Training-evaluation consistency (reward tied to position)
- C2: intrinsic_weight default raised to 0.1
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

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions."""
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
    """Deep Q-Network for discrete action space (buy/sell/hold)."""

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
        """Select action using epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()


class DQNTrainer:
    """
    DQN Trainer for financial trading.

    Args:
        ticker: Stock ticker symbol
        revise_state_func: Function to enhance raw state with LLM-generated features
        intrinsic_reward_func: Function to compute intrinsic reward
        state_dim: Dimension of enhanced state (WITHOUT position flag)
        intrinsic_weight: Weight for intrinsic reward (default: 0.1, was 0.02 in exp4.7)
        commission: Transaction cost rate (default: 0.001 = 0.1%)
    """

    def __init__(
        self,
        ticker: str,
        revise_state_func: Callable,
        intrinsic_reward_func: Callable,
        state_dim: int,
        intrinsic_weight: float = 0.1,
        commission: float = 0.001,
        device: str = None
    ):
        self.ticker = ticker
        self.revise_state = revise_state_func
        self.intrinsic_reward = intrinsic_reward_func
        self.intrinsic_weight = intrinsic_weight
        self.commission = commission

        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # B1: state_dim includes position flag (+1)
        self.feature_dim = state_dim  # dimension without position flag
        self.state_dim = state_dim + 1  # +1 for position flag

        # Networks (use state_dim which includes position flag)
        self.dqn = DQN(self.state_dim, device=self.device).to(self.device)
        self.target_dqn = DQN(self.state_dim, device=self.device).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()

        # Hyperparameters
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

        # 缓存加速特征计算 (only revise_state output, without position flag)
        self._feature_cache = {}

    def extract_state(
        self,
        data_loader,
        date: str,
        window: int = 20
    ) -> Optional[np.ndarray]:
        """
        Extract 20-day window of OHLCV data as raw state.

        Returns 120-dimensional array:
        - s[0:19]: 20 days close prices
        - s[20:39]: 20 days open prices
        - s[40:59]: 20 days high prices
        - s[60:79]: 20 days low prices
        - s[80:99]: 20 days volume
        - s[100:119]: 20 days adjusted close
        """
        dates = data_loader.get_date_range()
        try:
            idx = dates.index(date)
        except ValueError:
            return None

        if idx < window - 1:
            return None

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
                    # If price_dict is just a single value (adjusted_close)
                    state_120d.extend([price_dict] * 6)

        if len(state_120d) < 120:
            # Pad with zeros if not enough data
            state_120d.extend([0] * (120 - len(state_120d)))

        return np.array(state_120d[:120])

    def _get_cached_features(self, data_loader, date):
        """缓存加速特征计算（不含持仓标记）"""
        date_str = str(date)
        if date_str not in self._feature_cache:
            raw_state = self.extract_state(data_loader, date)
            if raw_state is not None:
                self._feature_cache[date_str] = self.revise_state(raw_state)
            else:
                self._feature_cache[date_str] = None
        return self._feature_cache[date_str]

    def _append_position_flag(self, features: np.ndarray, position: int) -> np.ndarray:
        """B1: Append position flag (0.0 or 1.0) to feature vector."""
        pos_flag = np.array([1.0 if position == 1 else 0.0])
        return np.concatenate([features, pos_flag])

    def train(
        self,
        train_data_loader,
        start_date: str,
        end_date: str,
        max_episodes: int = 100
    ) -> Dict:
        """
        Train DQN on training data.

        Changes from exp4.7:
        - C1: Reward is tied to position (0 reward when not holding)
        - B1: State includes position flag
        - B2: Transaction cost penalty applied on trade execution
        """
        dates = [
            d for d in train_data_loader.get_date_range()
            if start_date <= str(d) <= end_date
        ]

        logger.info(f"Training on {len(dates)} dates from {start_date} to {end_date}")
        print(f"Training on {len(dates)} dates from {start_date} to {end_date}")

        # 预计算所有特征以加速训练
        logger.info("预计算特征...")
        print("预计算特征...")
        for date in dates:
            self._get_cached_features(train_data_loader, date)
        logger.info(f"特征计算完成，缓存 {len(self._feature_cache)} 个状态")
        print(f"特征计算完成，缓存 {len(self._feature_cache)} 个状态")

        for episode in range(max_episodes):
            episode_reward = 0
            epsilon = max(self.epsilon_end, self.epsilon * (self.epsilon_decay ** episode))

            # C1+B1+B2: Track position throughout episode
            position = 0  # 0=no position, 1=holding
            trade_count = 0

            # 每个episode遍历所有日期
            for i, date in enumerate(dates):
                features = self._get_cached_features(train_data_loader, date)
                if features is None:
                    continue

                # B1: Append position flag to state
                state_with_pos = self._append_position_flag(features, position)

                # Select action
                action = self.dqn.select_action(state_with_pos, epsilon)

                current_price = train_data_loader.get_ticker_price_by_date(self.ticker, date)

                # C1+B2: Execute action and update position
                trade_executed = False
                if action == 0 and position == 0:  # Buy from empty
                    position = 1
                    trade_executed = True
                    trade_count += 1
                elif action == 1 and position == 1:  # Sell from holding
                    position = 0
                    trade_executed = True
                    trade_count += 1
                # action == 2 (Hold): no change

                # C1: Calculate reward tied to position
                if i < len(dates) - 1:
                    next_date = dates[i + 1]
                    next_price = train_data_loader.get_ticker_price_by_date(self.ticker, next_date)

                    # C1: Only earn return when holding position
                    if position == 1:
                        reward = (next_price - current_price) / current_price
                    else:
                        reward = 0.0

                    # B2: Transaction cost penalty
                    if trade_executed:
                        reward -= self.commission

                    # Add intrinsic reward (uses features WITHOUT position flag)
                    intrinsic_r = self.intrinsic_reward(features)
                    total_reward = reward + self.intrinsic_weight * intrinsic_r

                    # Get next state with position flag
                    next_features = self._get_cached_features(train_data_loader, next_date)
                    if next_features is not None:
                        next_state_with_pos = self._append_position_flag(next_features, position)
                    else:
                        next_state_with_pos = state_with_pos
                else:
                    # 最后一个日期，episode结束
                    next_state_with_pos = state_with_pos
                    total_reward = 0

                # Store experience (state includes position flag)
                self.buffer.push(state_with_pos, action, total_reward, next_state_with_pos, False)

                # Record for feature analysis (store features without position flag)
                self.episode_states.append(features.copy())
                self.episode_rewards.append(total_reward)

                # Train network
                if len(self.buffer) > self.batch_size:
                    self._update_network()

                episode_reward += total_reward

            # Soft update target network
            self._soft_update_target()

            # 每个episode都打印，强制刷新
            print(f"Episode {episode}/{max_episodes}, Epsilon: {epsilon:.3f}, "
                  f"Trades: {trade_count}, Buffer: {len(self.buffer)}", flush=True)
            logger.info(f"Episode {episode}/{max_episodes}, Epsilon: {epsilon:.3f}, "
                        f"Trades: {trade_count}, Buffer: {len(self.buffer)}")

        return self._get_training_summary()

    def _update_network(self) -> None:
        """Update Q-network using a batch of experiences."""
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
        """Soft update target network parameters."""
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _get_training_summary(self) -> Dict:
        """Return training summary for feature analysis."""
        return {
            'states': self.episode_states,
            'rewards': self.episode_rewards
        }

    def get_state_dict_portable(self) -> dict:
        """Get model state dict on CPU for cross-GPU serialization."""
        return {k: v.cpu() for k, v in self.dqn.state_dict().items()}

    def load_state_dict_portable(self, state_dict: dict):
        """Load model state dict from CPU tensors."""
        self.dqn.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def evaluate(
        self,
        val_data_loader,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Evaluate DQN on validation/test data.

        Now consistent with training: uses position flag in state,
        reward tied to position, includes commission.
        """
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

        for date in dates:
            raw_state = self.extract_state(val_data_loader, date)
            if raw_state is None:
                continue

            features = self.revise_state(raw_state)
            # B1: Append position flag
            state_with_pos = self._append_position_flag(features, current_position)
            action = self.dqn.select_action(state_with_pos, epsilon=0.0)

            current_price = val_data_loader.get_ticker_price_by_date(self.ticker, date)

            if action == 0:  # Buy
                if current_position == 0:
                    current_position = 1
                    entry_price = current_price
                    trades.append(('buy', date, current_price))
            elif action == 1:  # Sell
                if current_position == 1:
                    current_position = 0
                    trades.append(('sell', date, current_price))

            # Calculate daily return (consistent with training)
            if prev_price is not None:
                if current_position == 1:
                    daily_return = (current_price - prev_price) / prev_price
                else:
                    daily_return = 0.0
                daily_returns.append(daily_return)

            prev_price = current_price

        # Calculate metrics
        sharpe = self._calculate_sharpe(daily_returns)
        max_dd = self._calculate_max_drawdown(daily_returns)
        total_return_pct = sum(daily_returns) * 100

        return {
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_return': total_return_pct,
            'trades': trades,
            'num_trades': len([t for t in trades if t[0] in ('buy', 'sell')])
        }

    def _calculate_sharpe(self, daily_returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(daily_returns) < 2:
            return 0.0

        returns = np.array(daily_returns)
        mean_return = returns.mean() * 252  # Annualized
        std_return = returns.std() * np.sqrt(252)  # Annualized

        if std_return == 0:
            return 0.0

        return (mean_return - risk_free_rate) / std_return

    def _calculate_max_drawdown(self, daily_returns: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if len(daily_returns) < 2:
            return 0.0

        cumulative = np.cumsum(daily_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max)
        max_dd = abs(drawdowns.min()) * 100

        return max_dd
