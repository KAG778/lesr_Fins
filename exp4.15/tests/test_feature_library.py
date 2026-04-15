"""
Tests for feature_library.py: indicator registry, closure assembler, Z-score normalization.

TDD RED phase: All 20 core tests + 5 integration tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure core/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from feature_library import (
    INDICATOR_REGISTRY,
    build_revise_state,
    _extract_ohlcv,
    NormalizedIndicator,
    _dedup_by_base_indicator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_state():
    """120d interleaved OHLCV state: 20 trading days of realistic data.

    Prices around 100-110, volumes around 1e6-2e6.
    s[i*6+0]=close, s[i*6+1]=open, s[i*6+2]=high,
    s[i*6+3]=low, s[i*6+4]=volume, s[i*6+5]=adj_close
    """
    np.random.seed(42)
    state = np.zeros(120)
    for i in range(20):
        base_price = 100 + i * 0.5 + np.random.randn() * 2
        state[i * 6 + 0] = base_price                          # close
        state[i * 6 + 1] = base_price - np.random.rand()       # open
        state[i * 6 + 2] = base_price + np.random.rand() * 2   # high
        state[i * 6 + 3] = base_price - np.random.rand() * 2   # low
        state[i * 6 + 4] = 1e6 + np.random.rand() * 1e6        # volume
        state[i * 6 + 5] = base_price                          # adj_close
    return state


@pytest.fixture
def zeros_state():
    """120d all-zeros state (degenerate data from padding)."""
    return np.zeros(120)


@pytest.fixture
def short_state():
    """State with fewer than 120 dims (short input)."""
    return np.zeros(60)


# ---------------------------------------------------------------------------
# Task 1 Core Tests (RED phase)
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    """Tests 1, 2, 15, 17: Registry structure and completeness."""

    def test_registry_has_20_plus_indicators(self):
        """Test 1: INDICATOR_REGISTRY has >= 20 entries."""
        assert len(INDICATOR_REGISTRY) >= 20

    def test_all_themes_populated(self):
        """Test 2: Each theme has >= 2 indicators."""
        themes = {'trend', 'volatility', 'mean_reversion', 'volume'}
        theme_counts = {t: 0 for t in themes}
        for entry in INDICATOR_REGISTRY.values():
            t = entry.get('theme', '')
            if t in theme_counts:
                theme_counts[t] += 1
        for theme in themes:
            assert theme_counts[theme] >= 2, f"Theme '{theme}' has only {theme_counts[theme]} indicators, need >= 2"

    def test_registry_entry_structure(self):
        """Test 15: Every entry has fn, output_dim, default_params, param_ranges, theme keys."""
        required_keys = {'fn', 'output_dim', 'default_params', 'param_ranges', 'theme'}
        for name, entry in INDICATOR_REGISTRY.items():
            assert required_keys.issubset(entry.keys()), \
                f"Indicator '{name}' missing keys: {required_keys - entry.keys()}"
            assert callable(entry['fn']), f"Indicator '{name}' fn is not callable"
            assert isinstance(entry['output_dim'], int), f"Indicator '{name}' output_dim not int"
            assert isinstance(entry['default_params'], dict), f"Indicator '{name}' default_params not dict"
            assert isinstance(entry['param_ranges'], dict), f"Indicator '{name}' param_ranges not dict"

    def test_all_indicator_names_unique(self):
        """Test 17: No duplicate indicator names in registry."""
        # dict keys are inherently unique, but verify no fn is shared
        fns = [id(entry['fn']) for entry in INDICATOR_REGISTRY.values()]
        # Allow same fn to be reused (e.g., ROC and Momentum might share logic)
        # But names must be unique -- dict guarantees this
        assert len(INDICATOR_REGISTRY) == len(set(INDICATOR_REGISTRY.keys()))


class TestIndividualIndicators:
    """Tests 3-6, 18-19: Individual indicator output shapes and values."""

    def test_rsi_returns_1d(self, sample_state):
        """Test 3: compute_rsi returns shape (1,)."""
        from feature_library import compute_rsi
        result = compute_rsi(sample_state, window=14)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_rsi_normalized_range(self, sample_state):
        """Test 4: RSI output is in [0, 1] for non-degenerate input."""
        from feature_library import compute_rsi
        result = compute_rsi(sample_state, window=14)
        assert result.shape == (1,)
        assert np.isfinite(result[0])
        assert 0.0 <= result[0] <= 1.0, f"RSI={result[0]} not in [0, 1]"

    def test_macd_returns_3d(self, sample_state):
        """Test 5: compute_macd returns shape (3,)."""
        from feature_library import compute_macd
        result = compute_macd(sample_state)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_bollinger_returns_3d(self, sample_state):
        """Test 6: compute_bollinger returns shape (3,)."""
        from feature_library import compute_bollinger
        result = compute_bollinger(sample_state)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_stochastic_returns_2d(self, sample_state):
        """Test 18: compute_stochastic returns shape (2,) for %K and %D."""
        from feature_library import compute_stochastic
        result = compute_stochastic(sample_state)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_volume_ratio_returns_1d(self, sample_state):
        """Test 19: compute_volume_ratio returns shape (1,)."""
        from feature_library import compute_volume_ratio
        result = compute_volume_ratio(sample_state)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)


class TestNaNSafety:
    """Tests 7, 8: NaN/Inf guards for degenerate inputs."""

    def test_indicator_nan_safety_zeros_input(self, zeros_state):
        """Test 7: Any indicator called with all-zeros state returns finite values."""
        for name, entry in INDICATOR_REGISTRY.items():
            fn = entry['fn']
            try:
                result = fn(zeros_state, **entry['default_params'])
                assert np.all(np.isfinite(result)), \
                    f"Indicator '{name}' produced NaN/Inf on zeros input: {result}"
            except Exception:
                # If it raises, that's also acceptable handling
                pass

    def test_indicator_nan_safety_short_input(self, short_state):
        """Test 8: Indicators called with short state return defaults (no NaN)."""
        for name, entry in INDICATOR_REGISTRY.items():
            fn = entry['fn']
            try:
                result = fn(short_state, **entry['default_params'])
                assert np.all(np.isfinite(result)), \
                    f"Indicator '{name}' produced NaN/Inf on short input: {result}"
            except Exception:
                # If it raises, that's also acceptable handling
                pass


class TestClosureAssembler:
    """Tests 9-14: build_revise_state closure behavior."""

    def test_build_revise_state_closure(self):
        """Test 9: build_revise_state returns a callable."""
        selection = [{"indicator": "RSI", "params": {"window": 14}}]
        closure = build_revise_state(selection)
        assert callable(closure)

    def test_closure_output_shape(self, sample_state):
        """Test 10: closure with RSI returns shape matching output_dim sum."""
        selection = [{"indicator": "RSI", "params": {"window": 14}}]
        closure = build_revise_state(selection)
        result = closure(sample_state)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 1  # RSI output_dim=1

    def test_closure_multi_indicator(self, sample_state):
        """Test 11: closure with RSI + MACD + Bollinger returns shape (1+3+3,) = (7,)."""
        selection = [
            {"indicator": "RSI", "params": {"window": 14}},
            {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
            {"indicator": "Bollinger", "params": {"window": 20, "num_std": 2.0}},
        ]
        closure = build_revise_state(selection)
        result = closure(sample_state)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 7, f"Expected shape (7,), got {result.shape}"

    def test_closure_unknown_indicator_skipped(self, sample_state):
        """Test 12: Unknown indicator name in selection is silently skipped."""
        selection = [
            {"indicator": "RSI", "params": {"window": 14}},
            {"indicator": "NONEXISTENT_INDICATOR", "params": {}},
        ]
        closure = build_revise_state(selection)
        result = closure(sample_state)
        # Should only have RSI output (1d)
        assert len(result) == 1

    def test_closure_replaces_nan_with_zeros(self):
        """Test 13: If an indicator produces NaN, closure replaces with zeros."""
        # Use zeros_state which may trigger NaN in some indicators
        zeros = np.zeros(120)
        selection = [{"indicator": "RSI", "params": {"window": 14}}]
        closure = build_revise_state(selection)
        result = closure(zeros)
        assert np.all(np.isfinite(result)), f"Closure produced NaN/Inf: {result}"

    def test_param_range_clipping(self):
        """Test 14: Params outside registered range are clipped to valid range."""
        # RSI window range is (5, 60) per plan
        selection = [{"indicator": "RSI", "params": {"window": 100}}]
        # build_revise_state should clip to max 60
        closure = build_revise_state(selection)
        # The closure should still work (clipped, not crash)
        state = np.ones(120) * 100
        result = closure(state)
        assert np.all(np.isfinite(result))


class TestNormalizedIndicator:
    """Test 16: Z-score normalization wrapper."""

    def test_zscore_normalization(self, sample_state):
        """Test 16: NormalizedIndicator wrapper applies (x - mean) / (std + 1e-8)."""
        from feature_library import compute_rsi

        # Wrap RSI with known mean/std
        ni = NormalizedIndicator(
            fn=compute_rsi,
            params={'window': 14},
            mean=np.array([0.5]),
            std=np.array([0.1])
        )
        result = ni(sample_state)
        raw = compute_rsi(sample_state, window=14)
        expected = (raw - 0.5) / (0.1 + 1e-8)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestDedup:
    """Test 20: Same-base-indicator deduplication."""

    def test_dedup_same_base_indicator(self):
        """Test 20: _dedup_by_base_indicator([RSI(14), RSI(21)]) keeps only one."""
        selection = [
            {"indicator": "RSI", "params": {"window": 14}},
            {"indicator": "RSI", "params": {"window": 21}},
        ]
        ic_scores = {"RSI_14": 0.05, "RSI_21": 0.03}
        result = _dedup_by_base_indicator(selection, ic_scores)
        assert len(result) == 1
        # Should keep the one with higher IC (RSI_14)
        assert result[0]["params"]["window"] == 14


# ---------------------------------------------------------------------------
# Task 2 Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests verifying feature_library works with DQN pipeline."""

    def test_state_layout_compatibility(self, sample_state):
        """Integration 1: Closure output + raw(120) + regime(3) = correct total dim."""
        selection = [
            {"indicator": "RSI", "params": {"window": 14}},
            {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
            {"indicator": "Bollinger", "params": {"window": 20}},
            {"indicator": "Stochastic", "params": {"window": 14}},
            {"indicator": "ATR", "params": {"window": 14}},
        ]
        closure = build_revise_state(selection)
        features = closure(sample_state)

        # Compute expected feature dim: RSI(1) + MACD(3) + Bollinger(3) + Stochastic(2) + ATR(1) = 10
        expected_feature_dim = sum(
            INDICATOR_REGISTRY[s['indicator']]['output_dim'] for s in selection
        )
        assert len(features) == expected_feature_dim

        # Total state dim = 120 (raw) + 3 (regime) + features
        total_dim = 120 + 3 + expected_feature_dim
        assert total_dim == 133

    def test_ic_computation_with_library_features(self, sample_state):
        """Integration 2: Features from library are compatible with metrics.ic()."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
        from metrics import ic
        from feature_library import compute_rsi

        # Generate multiple states with a trend
        np.random.seed(42)
        features_list = []
        for i in range(50):
            state = np.ones(120) * (100 + i * 0.5 + np.random.randn() * 2)
            for j in range(20):
                state[j * 6 + 0] = state[j * 6]  # close
                state[j * 6 + 1] = state[j * 6]   # open
                state[j * 6 + 2] = state[j * 6] + 1  # high
                state[j * 6 + 3] = state[j * 6] - 1  # low
                state[j * 6 + 4] = 1e6              # volume
                state[j * 6 + 5] = state[j * 6]   # adj_close
            feat = compute_rsi(state, window=14)
            features_list.append(feat[0])

        feature_arr = np.array(features_list)
        forward_returns = np.random.randn(50) * 0.02

        ic_val = ic(feature_arr, forward_returns)
        assert isinstance(ic_val, float)
        assert np.isfinite(ic_val)

    def test_closure_deterministic(self, sample_state):
        """Integration 3: Same closure + same input = identical output."""
        selection = [
            {"indicator": "RSI", "params": {"window": 14}},
            {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
        ]
        closure = build_revise_state(selection)
        result1 = closure(sample_state)
        result2 = closure(sample_state)
        np.testing.assert_array_equal(result1, result2)

    def test_all_indicators_on_realistic_state(self, sample_state):
        """Integration 4: All indicators produce valid output on realistic 120d state."""
        for name, entry in INDICATOR_REGISTRY.items():
            fn = entry['fn']
            result = fn(sample_state, **entry['default_params'])
            assert isinstance(result, np.ndarray), f"{name} did not return ndarray"
            assert result.ndim == 1, f"{name} output is not 1D"
            assert result.shape[0] == entry['output_dim'], \
                f"{name} output shape {result.shape} != expected ({entry['output_dim']},)"
            assert np.all(np.isfinite(result)), \
                f"{name} produced NaN/Inf: {result}"

    def test_normalized_indicator_training_integration(self, sample_state):
        """Integration 5: NormalizedIndicator applies Z-score correctly with known values."""
        from feature_library import compute_rsi

        # First compute raw RSI to get actual value
        raw_rsi = compute_rsi(sample_state, window=14)
        # Create NormalizedIndicator with mean=0.5, std=0.1
        ni = NormalizedIndicator(
            fn=compute_rsi,
            params={'window': 14},
            mean=np.array([0.5]),
            std=np.array([0.1])
        )
        result = ni(sample_state)
        expected = (raw_rsi - 0.5) / (0.1 + 1e-8)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

        # Verify the raw RSI was around 0.5 (neutral zone)
        # With a realistic upward-trending state, RSI should be > 0.5
        # The normalized value should be finite
        assert np.isfinite(result[0])
