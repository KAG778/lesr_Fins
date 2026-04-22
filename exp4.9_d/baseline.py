"""
Baseline Strategy for exp4.9_d

- Uses feature_engine's revise_state (same features as LESR)
- B1: Position flag appended
- B2: Commission penalty
- D3: Reward is price change regardless of position (same as LESR)
- intrinsic_weight = 0 (no intrinsic reward)
"""

import sys
from pathlib import Path
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER' / 'backtest' / 'strategy' / 'timing_llm'))

from base_strategy_iso import BaseStrategyIso
from feature_engine import revise_state, STATE_DIM

logger = logging.getLogger(__name__)


def zero_intrinsic_reward(state: np.ndarray) -> float:
    return 0.0


class BaselineDQNStrategy(BaseStrategyIso):
    """Baseline DQN — same features as LESR but no intrinsic reward."""

    def __init__(self, ticker: str, trained_dqn, window: int = 20):
        super().__init__()
        self.ticker = ticker
        self.dqn = trained_dqn
        self.window = window
        self.current_position = 0

    def on_data(self, date, data_loader, framework):
        raw_state = self._extract_20day_window(date, data_loader)
        if raw_state is None:
            return

        features = revise_state(raw_state)
        pos_flag = np.array([1.0 if self.current_position == 1 else 0.0])
        state_with_pos = np.concatenate([features, pos_flag])

        try:
            action = self.dqn.select_action(state_with_pos, epsilon=0.0)
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return

        try:
            current_price = data_loader.get_ticker_price_by_date(self.ticker, date)
        except Exception as e:
            return

        if action == 0 and self.current_position == 0:
            framework.buy(date, self.ticker, current_price, -1)
            self.current_position = 1
        elif action == 1 and self.current_position == 1:
            if self.ticker in framework.portfolio:
                framework.sell(date, self.ticker, current_price,
                              framework.portfolio[self.ticker]['quantity'])
                self.current_position = 0

    def _extract_20day_window(self, date, data_loader) -> np.ndarray:
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
                        price_dict.get('close', 0), price_dict.get('open', 0),
                        price_dict.get('high', 0), price_dict.get('low', 0),
                        price_dict.get('volume', 0),
                        price_dict.get('adjusted_close', price_dict.get('close', 0))
                    ])
                else:
                    state_120d.extend([price_dict] * 6)

        if len(state_120d) < 120:
            state_120d.extend([0] * (120 - len(state_120d)))
        return np.array(state_120d[:120])


def train_baseline_dqn(
    ticker: str, data_loader, train_period: tuple, val_period: tuple,
    intrinsic_weight: float = 0.0, commission: float = 0.001
):
    """Train baseline DQN with same features but no intrinsic reward."""
    from dqn_trainer import DQNTrainer

    trainer = DQNTrainer(
        ticker=ticker,
        revise_state_func=revise_state,
        intrinsic_reward_func=zero_intrinsic_reward,
        state_dim=STATE_DIM,
        intrinsic_weight=intrinsic_weight,
        commission=commission
    )

    logger.info(f"Training baseline for {ticker}...")
    trainer.train(data_loader, train_period[0], train_period[1], max_episodes=50)
    val = trainer.evaluate(data_loader, val_period[0], val_period[1])
    logger.info(f"Baseline {ticker}: Sharpe={val['sharpe']:.3f} MaxDD={val['max_dd']:.2f}% Trades={val.get('num_trades',0)}")

    return trainer, val
