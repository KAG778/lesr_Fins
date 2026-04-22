"""
LESR Strategy Module for Exp4.9: Regime-Conditioned LESR

Changes from 4.7:
- on_data computes regime_vector and passes to revise_state
- Risk controller safety net (crisis override)
"""

import sys
from pathlib import Path
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso
from regime_detector import detect_regime, classify_regime

logger = logging.getLogger(__name__)


class LESRStrategy(BaseStrategyIso):
    """
    LESR Strategy with regime conditioning and risk safety net.
    """

    def __init__(
        self,
        ticker: str,
        revise_state_func: callable,
        trained_dqn,
        window: int = 20,
        enable_safety_net: bool = True
    ):
        super().__init__()
        self.ticker = ticker
        self.revise_state = revise_state_func
        self.dqn = trained_dqn
        self.window = window
        self.enable_safety_net = enable_safety_net

        # Track recent trades for safety net
        self._recent_trades = []  # list of (date, action, pnl)
        self._consecutive_losses = 0

        self.logger.info(f"Initialized LESRStrategy for {ticker} (safety_net={enable_safety_net})")

    def on_data(self, date, data_loader, framework):
        """Called for each trading day."""
        # Extract raw state
        raw_state = self._extract_20day_window(date, data_loader)
        if raw_state is None:
            return

        # Compute regime vector
        regime_vector = detect_regime(raw_state)
        regime_label = classify_regime(regime_vector)

        # Apply LLM feature enhancement WITH regime
        try:
            enhanced_state = self.revise_state(raw_state, regime_vector)
        except Exception as e:
            self.logger.error(f"Error in revise_state: {e}")
            return

        # DQN decision
        try:
            action = self.dqn.select_action(enhanced_state, epsilon=0.0)
        except Exception as e:
            self.logger.error(f"Error in DQN action selection: {e}")
            return

        # Risk safety net
        if self.enable_safety_net:
            action = self._apply_safety_net(action, regime_vector, enhanced_state, date)

        # Get current price
        try:
            current_price = data_loader.get_ticker_price_by_date(self.ticker, date)
        except Exception as e:
            self.logger.error(f"Error getting price for {self.ticker} on {date}: {e}")
            return

        # Execute trade
        if action == 0:  # Buy
            framework.buy(date, self.ticker, current_price, -1)
            self.logger.info(f"BUY {self.ticker} at {current_price:.2f} on {date} [{regime_label}]")

        elif action == 1:  # Sell
            if self.ticker in framework.portfolio:
                framework.sell(
                    date, self.ticker, current_price,
                    framework.portfolio[self.ticker]['quantity']
                )
                self.logger.info(f"SELL {self.ticker} at {current_price:.2f} on {date} [{regime_label}]")

    def _apply_safety_net(self, action: int, regime_vector: np.ndarray, 
                          enhanced_state: np.ndarray, date) -> int:
        """
        Hard constraint safety net. Overrides DQN action in dangerous conditions.
        
        Rules:
        1. crisis_signal > 0.7 + BUY → HOLD (don't enter during crisis)
        2. volatility_regime > 0.8 + BUY + no position → HOLD (avoid high vol entry)
        3. consecutive_losses >= 3 + BUY → HOLD (cooling period)
        """
        crisis_signal = regime_vector[4]
        volatility_regime = regime_vector[1]
        trend_strength = regime_vector[0]

        if action == 0:  # Only filter BUY actions
            # Rule 1: Crisis override
            if crisis_signal > 0.7:
                self.logger.info(f"  Safety net: HOLD (crisis_signal={crisis_signal:.2f})")
                return 2  # Force HOLD

            # Rule 2: Extreme volatility + counter-trend
            if volatility_regime > 0.85 and trend_strength < -0.5:
                self.logger.info(f"  Safety net: HOLD (high_vol={volatility_regime:.2f} + downtrend)")
                return 2

            # Rule 3: Consecutive losses cooling
            if self._consecutive_losses >= 3:
                self.logger.info(f"  Safety net: HOLD (consecutive_losses={self._consecutive_losses})")
                return 2

        return action

    def _extract_20day_window(self, date, data_loader) -> np.ndarray:
        """Extract 20-day OHLCV window as raw state."""
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

        if len(state_120d) < 120:
            state_120d.extend([0] * (120 - len(state_120d)))

        return np.array(state_120d[:120])
