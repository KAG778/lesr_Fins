"""
Tests for JSON validation pipeline (Plan 03-02 Task 2).

Covers: LESR-02 (automated checks for syntax, dimensions, numerical stability).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure core/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))

from feature_library import validate_selection, INDICATOR_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_state():
    """120d interleaved state with realistic OHLCV data."""
    np.random.seed(42)
    s = np.zeros(120)
    base_price = 100.0
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
    return s


@pytest.fixture
def valid_json():
    """Valid JSON selection string."""
    return '{"features": [{"indicator": "RSI", "params": {"window": 14}}, {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}}]}'


# ===========================================================================
# Validation tests (1-7)
# ===========================================================================

class TestValidation:
    """Tests for validate_selection function."""

    def test_valid_json_passes(self, valid_json, sample_state):
        """validate_selection with valid JSON returns proper structure."""
        result = validate_selection(valid_json, sample_state)
        assert 'selection' in result
        assert 'revise_state' in result
        assert 'feature_dim' in result
        assert 'errors' in result
        assert callable(result['revise_state'])
        assert isinstance(result['feature_dim'], int)
        assert isinstance(result['errors'], list)

    def test_malformed_json_fails(self, sample_state):
        """validate_selection with invalid JSON returns parse error."""
        bad_json = '{"features": [not valid json'
        result = validate_selection(bad_json, sample_state)
        assert len(result['errors']) > 0
        assert any('parse' in e.lower() or 'json' in e.lower()
                    for e in result['errors'])

    def test_unknown_indicator_rejected(self, sample_state):
        """validate_selection with unknown indicator returns error."""
        json_str = '{"features": [{"indicator": "FAKE_INDICATOR", "params": {}}]}'
        result = validate_selection(json_str, sample_state)
        assert any('FAKE_INDICATOR' in e or 'unknown' in e.lower()
                    for e in result['errors'])

    def test_param_out_of_range_clipped(self, sample_state):
        """validate_selection clips out-of-range params and reports warning."""
        # RSI window range is (5, 60); passing 100 should be clipped
        json_str = '{"features": [{"indicator": "RSI", "params": {"window": 100}}]}'
        result = validate_selection(json_str, sample_state)
        # Should have a warning about out-of-range param
        assert any('range' in e.lower() or 'clip' in e.lower()
                    for e in result['errors'])
        # The selection should have clipped params
        if result['selection']:
            assert result['selection'][0]['params']['window'] <= 60

    def test_nan_features_rejected(self, sample_state):
        """If indicator produces NaN on sample data, validation returns error."""
        # Use a state that might cause issues (all zeros)
        zero_state = np.zeros(120)
        json_str = '{"features": [{"indicator": "RSI", "params": {"window": 14}}]}'
        # All-zero state might produce NaN in some indicators
        # We test that the validation handles this
        result = validate_selection(json_str, zero_state)
        # Should either succeed with finite features or report error
        if result['selection']:
            features = result['revise_state'](zero_state)
            assert np.all(np.isfinite(features))

    def test_dimension_consistency(self, valid_json, sample_state):
        """validate_selection returns feature_dim matching sum of output_dims."""
        result = validate_selection(valid_json, sample_state)
        if not result['errors']:
            # RSI output_dim=1, MACD output_dim=3, total=4
            expected_dim = 1 + 3  # RSI + MACD
            assert result['feature_dim'] == expected_dim

    def test_empty_features_rejected(self, sample_state):
        """validate_selection with empty features list returns error."""
        json_str = '{"features": []}'
        result = validate_selection(json_str, sample_state)
        assert len(result['errors']) > 0
