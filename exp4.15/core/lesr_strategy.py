"""
LESR Strategy Module for Exp4.7 Financial Trading Experiment

This module implements the LESR strategy that integrates with FINSABER framework
for backtesting. It uses a trained DQN model with LLM-generated features.
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso

logger = logging.getLogger(__name__)


class LESRStrategy(BaseStrategyIso):
    """
    LESR Strategy for FINSABER backtesting.

    This strategy uses a trained DQN model with LLM-generated features
    to make buy/sell/hold decisions.

    Args:
        ticker: Stock ticker symbol
        revise_state_func: Function to enhance raw state with features
        trained_dqn: Trained DQN model
        window: Lookback window size (default: 20 days)
    """

    def __init__(
        self,
        ticker: str,
        revise_state_func: callable,
        trained_dqn,
        window: int = 20
    ):
        super().__init__()
        self.ticker = ticker
        self.revise_state = revise_state_func
        self.dqn = trained_dqn
        self.window = window

        self.logger.info(f"Initialized LESRStrategy for {ticker}")

    def on_data(self, date, data_loader, framework):
        """
        Called for each trading day with new data.

        Args:
            date: Current date
            data_loader: Data loader instance
            framework: FINSABER framework instance
        """
        # Extract 20-day historical data
        raw_state = self._extract_20day_window(date, data_loader)
        if raw_state is None:
            return

        # Apply LLM feature enhancement
        try:
            enhanced_state = self.revise_state(raw_state)
        except Exception as e:
            self.logger.error(f"Error in revise_state: {e}")
            return

        # DQN decision (evaluation mode, no exploration)
        try:
            action = self.dqn.select_action(enhanced_state, epsilon=0.0)
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
            self.logger.info(f"BUY {self.ticker} at {current_price:.2f} on {date}")

        elif action == 1:  # Sell
            if self.ticker in framework.portfolio:
                framework.sell(
                    date,
                    self.ticker,
                    current_price,
                    framework.portfolio[self.ticker]['quantity']
                )
                self.logger.info(f"SELL {self.ticker} at {current_price:.2f} on {date}")

        # action == 2: Hold, do nothing

    def _extract_20day_window(self, date, data_loader) -> np.ndarray:
        """
        Extract 20-day OHLCV window as raw state.

        Returns:
            120-dimensional numpy array or None if insufficient data
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
                    # Single value (adjusted_close)
                    state_120d.extend([price_dict] * 6)

        # Pad if needed
        if len(state_120d) < 120:
            state_120d.extend([0] * (120 - len(state_120d)))

        return np.array(state_120d[:120])
