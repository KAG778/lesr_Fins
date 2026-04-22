"""
Baseline Strategy Module for Exp4.9_b Financial Trading Experiment

Changes from exp4.7:
- B1: Baseline also uses position flag (state_dim = 120 + 1 = 121)
- C1+B2: Baseline trains with position-aware reward and commission penalty
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add direct module paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER' / 'backtest' / 'strategy' / 'timing_llm'))

from base_strategy_iso import BaseStrategyIso
from dqn_trainer import DQN

logger = logging.getLogger(__name__)


def identity_revise_state(raw_state: np.ndarray) -> np.ndarray:
    """
    Identity function - no feature engineering.
    Returns raw state as-is for baseline comparison.
    """
    return raw_state


def zero_intrinsic_reward(state: np.ndarray) -> float:
    """
    Zero intrinsic reward - no intrinsic signal.
    Used for baseline to isolate extrinsic reward effect.
    """
    return 0.0


class BaselineDQNStrategy(BaseStrategyIso):
    """
    Baseline DQN Strategy using raw OHLCV data without feature engineering.

    Changes from exp4.7:
    - Tracks position state and appends position flag (consistent with LESR)
    """

    def __init__(
        self,
        ticker: str,
        trained_dqn,
        window: int = 20
    ):
        super().__init__()
        self.ticker = ticker
        self.dqn = trained_dqn
        self.window = window
        self.current_position = 0  # Track position

        self.logger.info(f"Initialized BaselineDQNStrategy for {ticker}")

    def on_data(self, date, data_loader, framework):
        """
        Called for each trading day with new data.
        """
        raw_state = self._extract_20day_window(date, data_loader)
        if raw_state is None:
            return

        # B1: Append position flag (consistent with training)
        pos_flag = np.array([1.0 if self.current_position == 1 else 0.0])
        state_with_pos = np.concatenate([raw_state, pos_flag])

        # Direct DQN decision
        try:
            action = self.dqn.select_action(state_with_pos, epsilon=0.0)
        except Exception as e:
            self.logger.error(f"Error in DQN action selection: {e}")
            return

        # Get current price
        try:
            current_price = data_loader.get_ticker_price_by_date(self.ticker, date)
        except Exception as e:
            self.logger.error(f"Error getting price for {self.ticker} on {date}: {e}")
            return

        # Execute trade (consistent with training)
        if action == 0 and self.current_position == 0:  # Buy from empty
            framework.buy(date, self.ticker, current_price, -1)
            self.current_position = 1

        elif action == 1 and self.current_position == 1:  # Sell from holding
            if self.ticker in framework.portfolio:
                framework.sell(
                    date,
                    self.ticker,
                    current_price,
                    framework.portfolio[self.ticker]['quantity']
                )
                self.current_position = 0

        # action == 2: Hold

    def _extract_20day_window(self, date, data_loader) -> np.ndarray:
        """
        Extract 20-day OHLCV window as raw state.
        """
        dates = data_loader.get_date_range()
        try:
            idx = dates.index(date)
        except ValueError:
            return None

        if idx < self.window - 1:
            return None

        window_dates = dates[idx - self.window + 1:idx + 1]
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

        # Pad if needed
        if len(state_120d) < 120:
            state_120d.extend([0] * (120 - len(state_120d)))

        return np.array(state_120d[:120])


def train_baseline_dqn(
    ticker: str,
    data_loader,
    train_period: tuple,
    val_period: tuple,
    state_dim: int = 120,
    intrinsic_weight: float = 0.0,
    commission: float = 0.001
):
    """
    Train a baseline DQN without feature engineering.

    Args:
        ticker: Stock ticker symbol
        data_loader: Data loader instance
        train_period: Training period (start_date, end_date)
        val_period: Validation period (start_date, end_date)
        state_dim: State dimension (default: 120 for raw OHLCV, DQNTrainer adds +1)
        intrinsic_weight: Intrinsic reward weight (default: 0 for baseline)
        commission: Transaction cost rate

    Returns:
        Tuple of (trainer, validation_metrics)
    """
    from dqn_trainer import DQNTrainer

    trainer = DQNTrainer(
        ticker=ticker,
        revise_state_func=identity_revise_state,
        intrinsic_reward_func=zero_intrinsic_reward,
        state_dim=state_dim,
        intrinsic_weight=intrinsic_weight,
        commission=commission
    )

    # Train
    logger.info(f"Training baseline DQN for {ticker}...")
    trainer.train(
        data_loader,
        train_period[0],
        train_period[1],
        max_episodes=50
    )

    # Evaluate
    logger.info(f"Evaluating baseline DQN for {ticker}...")
    val_metrics = trainer.evaluate(
        data_loader,
        val_period[0],
        val_period[1]
    )

    logger.info(f"Baseline Sharpe: {val_metrics['sharpe']:.3f}, MaxDD: {val_metrics['max_dd']:.2f}%, Trades: {val_metrics.get('num_trades', 0)}")

    return trainer, val_metrics
