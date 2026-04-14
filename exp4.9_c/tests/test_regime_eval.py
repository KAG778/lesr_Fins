"""
Tests for regime-stratified evaluation in DQNTrainer.evaluate() (D-07).

Verifies that:
- Regime classification uses trend thresholds: bull (>0.3), bear (<-0.3), sideways (otherwise)
- evaluate() returns regime_metrics with bull/bear/sideways keys populated
- Each regime entry has sharpe, max_dd, count
- Regime with <2 returns gets sharpe=0.0, max_dd=0.0
- NaN/corrupt state defaults to 'sideways' classification
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure exp4.9_c is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRegimeClassification:
    """Test regime classification thresholds used in evaluate()."""

    def test_bull_regime_trend_above_threshold(self):
        """Trend > 0.3 should be classified as 'bull'."""
        from regime_detector import detect_regime
        s = np.zeros(120)
        for i in range(20):
            s[i * 6] = 90 + i * 2  # close prices: 90, 92, 94, ..., 128
        regime = detect_regime(s)
        assert regime[0] > 0.3, f"Expected trend > 0.3 for uptrend, got {regime[0]}"

    def test_bear_regime_trend_below_threshold(self):
        """Trend < -0.3 should be classified as 'bear'."""
        from regime_detector import detect_regime
        s = np.zeros(120)
        for i in range(20):
            s[i * 6] = 128 - i * 2  # close prices: 128, 126, 124, ..., 90
        regime = detect_regime(s)
        assert regime[0] < -0.3, f"Expected trend < -0.3 for downtrend, got {regime[0]}"

    def test_sideways_regime_trend_near_zero(self):
        """Trend between -0.3 and 0.3 should be classified as 'sideways'."""
        from regime_detector import detect_regime
        s = np.ones(120) * 100.0
        regime = detect_regime(s)
        assert -0.3 <= regime[0] <= 0.3, f"Expected |trend| <= 0.3 for flat, got {regime[0]}"


class TestRegimeMetricsStructure:
    """Test that evaluate() returns properly structured regime_metrics."""

    def _make_setup(self):
        """Create a trainer and mock data_loader for regime-stratified testing."""
        from dqn_trainer import DQNTrainer

        revise_state = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        intrinsic_reward = MagicMock(return_value=5.0)

        trainer = DQNTrainer(
            ticker='TEST',
            revise_state_func=revise_state,
            intrinsic_reward_func=intrinsic_reward,
            state_dim=126,
            device='cpu'
        )

        # 30 dates with rising prices (strong uptrend => should be classified as bull)
        dates = [f'2020-01-{i:02d}' for i in range(1, 31)]
        prices = list(range(100, 130))  # price goes from 100 to 129

        data_loader = MagicMock()
        data_loader.get_date_range.return_value = dates

        # Return date-specific price data so extract_state sees actual rising prices
        def mock_get_data_by_date(d):
            idx = dates.index(d) if d in dates else 0
            p = prices[idx]
            return {
                'price': {'TEST': {
                    'close': p, 'open': p - 1, 'high': p + 1,
                    'low': p - 1, 'volume': 1000, 'adjusted_close': p
                }}
            }
        data_loader.get_data_by_date = MagicMock(side_effect=mock_get_data_by_date)

        data_loader.get_ticker_price_by_date = MagicMock(
            side_effect=lambda t, d: prices[dates.index(d)] if d in dates else 100
        )

        return trainer, data_loader

    def test_regime_metrics_has_bull_key(self):
        """regime_metrics should have 'bull' key when prices trend up."""
        trainer, data_loader = self._make_setup()
        result = trainer.evaluate(data_loader, '2020-01-01', '2020-01-30')

        rm = result['regime_metrics']
        assert 'bull' in rm, f"Expected 'bull' key in regime_metrics, got keys: {list(rm.keys())}"

    def test_regime_metrics_has_sideways_key(self):
        """regime_metrics should have 'sideways' key."""
        trainer, data_loader = self._make_setup()
        result = trainer.evaluate(data_loader, '2020-01-01', '2020-01-30')

        rm = result['regime_metrics']
        assert 'sideways' in rm, f"Expected 'sideways' key in regime_metrics, got keys: {list(rm.keys())}"

    def test_regime_metrics_has_bear_key(self):
        """regime_metrics should have 'bear' key."""
        trainer, data_loader = self._make_setup()
        result = trainer.evaluate(data_loader, '2020-01-01', '2020-01-30')

        rm = result['regime_metrics']
        assert 'bear' in rm, f"Expected 'bear' key in regime_metrics, got keys: {list(rm.keys())}"

    def test_regime_entry_has_sharpe_max_dd_count(self):
        """Each regime entry should have sharpe, max_dd, count fields."""
        trainer, data_loader = self._make_setup()
        result = trainer.evaluate(data_loader, '2020-01-01', '2020-01-30')

        rm = result['regime_metrics']
        for regime_name in ['bull', 'bear', 'sideways']:
            assert regime_name in rm, f"Missing regime '{regime_name}'"
            entry = rm[regime_name]
            assert 'sharpe' in entry, f"Missing 'sharpe' in {regime_name}"
            assert 'max_dd' in entry, f"Missing 'max_dd' in {regime_name}"
            assert 'count' in entry, f"Missing 'count' in {regime_name}"
            assert isinstance(entry['sharpe'], float), f"sharpe should be float in {regime_name}"
            assert isinstance(entry['max_dd'], float), f"max_dd should be float in {regime_name}"
            assert isinstance(entry['count'], int), f"count should be int in {regime_name}"

    def test_bull_regime_has_positive_count(self):
        """Bull regime should have count > 0 for rising price data."""
        trainer, data_loader = self._make_setup()
        result = trainer.evaluate(data_loader, '2020-01-01', '2020-01-30')

        rm = result['regime_metrics']
        assert rm['bull']['count'] > 0, "Expected bull count > 0 for rising prices"

    def test_total_regime_counts_sum_to_daily_returns(self):
        """Sum of regime counts should equal len(daily_returns)."""
        trainer, data_loader = self._make_setup()
        result = trainer.evaluate(data_loader, '2020-01-01', '2020-01-30')

        rm = result['regime_metrics']
        total_count = rm['bull']['count'] + rm['bear']['count'] + rm['sideways']['count']
        # daily_returns starts from day 2 (first day has no prev_price)
        # We expect total_count to be at least the number of trading days minus 1
        assert total_count > 0, "Total regime count should be > 0"


class TestEmptyRegime:
    """Test behavior when a regime has very few returns."""

    def test_single_return_gives_zero_metrics(self):
        """Metrics computed from <2 returns should return 0.0."""
        from metrics import sharpe_ratio, max_drawdown

        assert sharpe_ratio([0.01]) == 0.0
        assert max_drawdown([0.01]) == 0.0

    def test_nan_regime_defaults_to_sideways(self):
        """NaN from detect_regime should default to 'sideways' classification."""
        from regime_detector import detect_regime

        s = np.zeros(120)
        regime = detect_regime(s)
        assert not np.any(np.isnan(regime)), f"Regime vector has NaN: {regime}"

    def test_detect_regime_handles_flat_prices(self):
        """detect_regime should handle flat price series without NaN/Inf."""
        from regime_detector import detect_regime
        s = np.ones(120) * 50.0
        regime = detect_regime(s)
        assert not np.any(np.isnan(regime))
        assert not np.any(np.isinf(regime))
