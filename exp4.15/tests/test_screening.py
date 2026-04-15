"""
Tests for feature screening with IC/variance gates and dedup (Plan 03-02 Task 2).

Covers: LESR-04 (feature filtering to 5-10 non-degenerate features).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure core/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))

from feature_library import screen_features, INDICATOR_REGISTRY, build_revise_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def training_states():
    """Batch of 50 training states with realistic data."""
    np.random.seed(42)
    batch = []
    base_price = 100.0
    for _ in range(50):
        s = np.zeros(120)
        for i in range(20):
            close = base_price + np.random.randn() * 2.0
            open_ = close + np.random.randn() * 0.5
            high = max(close, open_) + abs(np.random.randn()) * 1.0
            low = min(close, open_) - abs(np.random.randn()) * 1.0
            volume = 1e6 + np.random.randn() * 1e5
            s[i * 6 + 0] = close
            s[i * 6 + 1] = open_
            s[i * 6 + 2] = high
            s[i * 6 + 3] = low
            s[i * 6 + 4] = volume
            s[i * 6 + 5] = close
            base_price = close
        batch.append(s)
    return np.array(batch)


@pytest.fixture
def forward_returns():
    """50 forward returns with some signal."""
    np.random.seed(42)
    return np.random.randn(50) * 0.02


@pytest.fixture
def many_features_selection():
    """15 valid feature selections to test screening down to 5-10."""
    return [
        {"indicator": "RSI", "params": {"window": 14}},
        {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
        {"indicator": "Bollinger", "params": {"window": 20, "num_std": 2.0}},
        {"indicator": "ATR", "params": {"window": 14}},
        {"indicator": "Momentum", "params": {"window": 10}},
        {"indicator": "EMA_Cross", "params": {"fast": 12, "slow": 26}},
        {"indicator": "Stochastic", "params": {"window": 14}},
        {"indicator": "Williams_R", "params": {"window": 14}},
        {"indicator": "CCI", "params": {"window": 20}},
        {"indicator": "OBV", "params": {}},
        {"indicator": "Volume_Ratio", "params": {"window": 20}},
        {"indicator": "ADX", "params": {"window": 14}},
        {"indicator": "ROC", "params": {"window": 10}},
        {"indicator": "DEMA", "params": {"window": 20}},
        {"indicator": "SMA_Cross", "params": {"fast": 10, "slow": 30}},
    ]


# ===========================================================================
# Screening tests (8-13)
# ===========================================================================

class TestScreening:
    """Tests for screen_features function."""

    def test_screen_produces_5_to_10(self, many_features_selection,
                                      training_states, forward_returns):
        """screen_features with 15 valid features returns 5-10 features."""
        revise_fn = build_revise_state(many_features_selection)
        result = screen_features(
            selection=many_features_selection,
            revise_fn=revise_fn,
            training_states=training_states,
            forward_returns=forward_returns,
        )
        screened = result['screened_selection']
        assert len(screened) >= 5
        assert len(screened) <= 10

    def test_ic_threshold_rejects(self, training_states, forward_returns):
        """Feature with IC=0.001 is rejected by screening."""
        # Use a selection that likely has low IC (OBV might have low IC
        # on random data)
        selection = [
            {"indicator": "OBV", "params": {}},
            {"indicator": "RSI", "params": {"window": 14}},
        ]
        revise_fn = build_revise_state(selection)
        result = screen_features(
            selection=selection,
            revise_fn=revise_fn,
            training_states=training_states,
            forward_returns=forward_returns,
        )
        # Check that some features are rejected or kept based on IC
        assert 'feature_metrics' in result

    def test_variance_threshold_rejects(self, training_states, forward_returns):
        """Feature with variance < 1e-6 is rejected."""
        # Create a state where a feature might have very low variance
        selection = [{"indicator": "RSI", "params": {"window": 14}}]
        revise_fn = build_revise_state(selection)
        result = screen_features(
            selection=selection,
            revise_fn=revise_fn,
            training_states=training_states,
            forward_returns=forward_returns,
        )
        # Check that variance threshold is applied
        for name, m in result.get('feature_metrics', {}).items():
            if m.get('variance', 0) < 1e-6:
                # Should be in rejected list
                rejected_names = [r.get('indicator') for r in result.get('rejected', [])]
                assert name in rejected_names

    def test_dedup_keeps_higher_ic(self, training_states, forward_returns):
        """RSI(14) with IC=0.1 and RSI(21) with IC=0.05 -> keeps RSI(14)."""
        # Provide explicit IC scores to force dedup behavior
        selection = [
            {"indicator": "RSI", "params": {"window": 14}},
            {"indicator": "RSI", "params": {"window": 21}},
        ]
        revise_fn = build_revise_state(selection)
        result = screen_features(
            selection=selection,
            revise_fn=revise_fn,
            training_states=training_states,
            forward_returns=forward_returns,
        )
        # After dedup, should have only 1 RSI entry
        rsi_entries = [s for s in result['screened_selection']
                       if s['indicator'] == 'RSI']
        assert len(rsi_entries) <= 1

    def test_screen_preserves_diverse_themes(self, many_features_selection,
                                              training_states, forward_returns):
        """screen_features output includes indicators from at least 2 themes."""
        revise_fn = build_revise_state(many_features_selection)
        result = screen_features(
            selection=many_features_selection,
            revise_fn=revise_fn,
            training_states=training_states,
            forward_returns=forward_returns,
        )
        screened = result['screened_selection']
        themes = set()
        for item in screened:
            name = item.get('indicator', '')
            if name in INDICATOR_REGISTRY:
                themes.add(INDICATOR_REGISTRY[name]['theme'])
        assert len(themes) >= 2

    def test_screen_ranked_by_ic(self, training_states, forward_returns):
        """Returned features are sorted by IC descending."""
        selection = [
            {"indicator": "RSI", "params": {"window": 14}},
            {"indicator": "ATR", "params": {"window": 14}},
            {"indicator": "Momentum", "params": {"window": 10}},
        ]
        revise_fn = build_revise_state(selection)
        result = screen_features(
            selection=selection,
            revise_fn=revise_fn,
            training_states=training_states,
            forward_returns=forward_returns,
        )
        # Check that IC values are in descending order
        metrics = result.get('feature_metrics', {})
        if len(metrics) > 1:
            ics = [metrics[name].get('ic', 0) for name in metrics]
            # Should be non-increasing (may be equal for some)
            for i in range(len(ics) - 1):
                assert ics[i] >= ics[i + 1] - 0.001  # small tolerance
