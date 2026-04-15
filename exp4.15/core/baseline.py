"""
Baseline Strategy Module for Exp4.7 Financial Trading Experiment

This module implements baseline strategies for comparison:
1. Baseline DQN: Uses raw OHLCV data without feature engineering
2. Baseline MLP: Same as Baseline DQN but with simpler architecture
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add direct module paths for imports
sys.path.insert(0, str(Path(__file__).parent))  # core/ for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'FINSABER'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'FINSABER' / 'backtest' / 'strategy' / 'timing_llm'))

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

    This serves as the baseline to compare against LESR's enhanced features.
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

        self.logger.info(f"Initialized BaselineDQNStrategy for {ticker}")

    def on_data(self, date, data_loader, framework):
        """
        Called for each trading day with new data.
        """
        # Extract 20-day historical data (no feature engineering)
        raw_state = self._extract_20day_window(date, data_loader)
        if raw_state is None:
            return

        # Direct DQN decision on raw state
        try:
            action = self.dqn.select_action(raw_state, epsilon=0.0)
        except Exception as e:
            self.logger.error(f"Error in DQN action selection: {e}")
            return

        # Get current price
        try:
            current_price = data_loader.get_ticker_price_by_date(self.ticker, date)
        except Exception as e:
            self.logger.error(f"Error getting price for {self.ticker} on {date}: {e}")
            return

        # Execute trade
        if action == 0:  # Buy
            framework.buy(date, self.ticker, current_price, -1)

        elif action == 1:  # Sell
            if self.ticker in framework.portfolio:
                framework.sell(
                    date,
                    self.ticker,
                    current_price,
                    framework.portfolio[self.ticker]['quantity']
                )

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
    intrinsic_weight: float = 0.0
):
    """
    Train a baseline DQN without feature engineering.

    Args:
        ticker: Stock ticker symbol
        data_loader: Data loader instance
        train_period: Training period (start_date, end_date)
        val_period: Validation period (start_date, end_date)
        state_dim: State dimension (default: 120 for raw OHLCV)
        intrinsic_weight: Intrinsic reward weight (default: 0 for baseline)

    Returns:
        Tuple of (trainer, validation_metrics)
    """
    from dqn_trainer import DQNTrainer

    trainer = DQNTrainer(
        ticker=ticker,
        revise_state_func=identity_revise_state,
        intrinsic_reward_func=zero_intrinsic_reward,
        state_dim=state_dim,
        intrinsic_weight=intrinsic_weight
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

    logger.info(f"Baseline Sharpe: {val_metrics['sharpe']:.3f}, MaxDD: {val_metrics['max_dd']:.2f}%")

    return trainer, val_metrics
