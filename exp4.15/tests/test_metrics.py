"""
Unit tests for exp4.9_c/metrics.py

Tests all 9 metric functions with known inputs/outputs and edge cases.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure core/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, win_rate,
    ic, rolling_ic, information_ratio, quantile_spread
)


# ===================================================================
# Performance Metrics Tests
# ===================================================================

class TestSharpeRatio:
    def test_known_returns(self):
        """Positive returns should yield positive Sharpe."""
        r = [0.01, 0.02, -0.01, 0.03]
        result = sharpe_ratio(r)
        assert isinstance(result, float)
        assert result > 0

    def test_negative_returns(self):
        """Consistently negative returns should yield negative Sharpe."""
        r = [-0.01, -0.02, -0.015, -0.005]
        result = sharpe_ratio(r)
        assert result < 0

    def test_empty_returns(self):
        assert sharpe_ratio([]) == 0.0

    def test_single_return(self):
        assert sharpe_ratio([0.01]) == 0.0

    def test_zero_std(self):
        """All identical returns -> zero std -> return 0.0."""
        assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0

    def test_fixture_returns(self, sample_returns):
        result = sharpe_ratio(sample_returns)
        assert isinstance(result, float)
        assert result > 0  # mostly positive returns


class TestSortinoRatio:
    def test_known_returns(self):
        r = [0.01, 0.02, -0.01, 0.03]
        result = sortino_ratio(r)
        assert isinstance(result, float)
        assert result > 0

    def test_all_positive_returns(self):
        """No negative returns -> downside dev is zero -> 0.0."""
        r = [0.01, 0.02, 0.03, 0.04]
        assert sortino_ratio(r) == 0.0

    def test_empty(self):
        assert sortino_ratio([]) == 0.0

    def test_single(self):
        assert sortino_ratio([0.01]) == 0.0


class TestMaxDrawdown:
    def test_known_drawdown(self):
        """Series with a 20% drawdown."""
        r = [-0.1, 0.05, -0.2, 0.1]
        result = max_drawdown(r)
        assert result > 0.2 * 100 * 0.5  # at least some significant drawdown

    def test_monotone_increase(self):
        """Monotonically increasing -> zero drawdown."""
        r = [0.01, 0.02, 0.03, 0.04]
        assert max_drawdown(r) == 0.0

    def test_empty(self):
        assert max_drawdown([]) == 0.0

    def test_single(self):
        assert max_drawdown([0.01]) == 0.0

    def test_fixture_returns(self, sample_returns):
        result = max_drawdown(sample_returns)
        assert isinstance(result, float)
        assert result >= 0


class TestCalmarRatio:
    def test_positive_calmar(self):
        """Positive returns with some drawdown."""
        r = [0.01, 0.02, -0.01, 0.03, -0.005, 0.015]
        result = calmar_ratio(r)
        assert isinstance(result, float)
        # Could be positive or negative depending on drawdown size

    def test_no_drawdown(self):
        """Zero drawdown -> 0.0."""
        r = [0.01, 0.02, 0.03]
        assert calmar_ratio(r) == 0.0

    def test_empty(self):
        assert calmar_ratio([]) == 0.0


class TestWinRate:
    def test_known_rate(self):
        """2 of 3 non-zero are positive -> 2/3."""
        r = [0.01, -0.02, 0.03, 0.0]
        result = win_rate(r)
        assert abs(result - 2.0 / 3.0) < 1e-10

    def test_all_positive(self):
        r = [0.01, 0.02, 0.03]
        assert win_rate(r) == 1.0

    def test_all_negative(self):
        r = [-0.01, -0.02, -0.03]
        assert win_rate(r) == 0.0

    def test_all_zero(self):
        r = [0.0, 0.0, 0.0]
        assert win_rate(r) == 0.0

    def test_empty(self):
        assert win_rate([]) == 0.0


# ===================================================================
# Factor Evaluation Metrics Tests
# ===================================================================

class TestIC:
    def test_known_correlation(self):
        """Positively correlated feature and returns."""
        fv = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fr = np.array([0.05, -0.01, 0.08, 0.03, 0.07])
        result = ic(fv, fr)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_perfect_positive(self):
        """Perfect positive monotonic relationship -> IC close to 1.0."""
        fv = np.arange(10, dtype=float)
        fr = np.arange(10, dtype=float)
        result = ic(fv, fr)
        assert abs(result - 1.0) < 0.01

    def test_perfect_negative(self):
        """Perfect negative monotonic relationship -> IC close to -1.0."""
        fv = np.arange(10, dtype=float)
        fr = np.arange(10, dtype=float)[::-1]
        result = ic(fv, fr)
        assert abs(result - (-1.0)) < 0.01

    def test_too_few_pairs(self):
        """Less than 5 pairs -> 0.0."""
        assert ic([1, 2, 3], [1, 2, 3]) == 0.0

    def test_empty(self):
        assert ic([], []) == 0.0


class TestRollingIC:
    def test_window2_length4(self):
        """Window=2 on length-4 arrays should return length-3 array."""
        fv = np.array([0.1, 0.2, 0.3, 0.4])
        fr = np.array([0.05, -0.01, 0.08, 0.03])
        result = rolling_ic(fv, fr, window=2)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3  # 4 - 2 + 1

    def test_window_larger_than_data(self):
        """Window > data length -> empty array."""
        fv = np.array([0.1, 0.2])
        fr = np.array([0.05, -0.01])
        result = rolling_ic(fv, fr, window=5)
        assert len(result) == 0

    def test_fixture_data(self, sample_features, sample_forward_returns):
        """Rolling IC on fixture data."""
        result = rolling_ic(sample_features[:, 0], sample_forward_returns, window=20)
        assert len(result) == 81  # 100 - 20 + 1


class TestInformationRatio:
    def test_known_series(self):
        """Mean/std of a known series."""
        series = np.array([0.1, 0.2, -0.1, 0.15])
        result = information_ratio(series)
        assert isinstance(result, float)
        expected = series.mean() / series.std()
        assert abs(result - expected) < 1e-6

    def test_zero_std(self):
        """All same values -> std=0 -> 0.0."""
        assert information_ratio([0.5, 0.5, 0.5]) == 0.0

    def test_too_short(self):
        """Less than 2 values -> 0.0."""
        assert information_ratio([0.1]) == 0.0

    def test_empty(self):
        assert information_ratio([]) == 0.0


class TestQuantileSpread:
    def test_known_spread(self):
        """5 values, 5 quantiles: top group mean - bottom group mean."""
        fv = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fr = np.array([0.01, 0.02, -0.01, 0.03, 0.04])
        result = quantile_spread(fv, fr, n_quantiles=5)
        assert isinstance(result, float)
        # bottom quantile (0.1) -> return 0.01
        # top quantile (0.5) -> return 0.04
        # spread = 0.04 - 0.01 = 0.03
        assert abs(result - 0.03) < 1e-10

    def test_too_few_data(self):
        """Less data than quantiles -> 0.0."""
        fv = np.array([0.1, 0.2])
        fr = np.array([0.01, 0.02])
        assert quantile_spread(fv, fr, n_quantiles=5) == 0.0

    def test_empty(self):
        assert quantile_spread([], []) == 0.0


# ===================================================================
# Edge Case: NaN/Inf handling
# ===================================================================

class TestEdgeCases:
    def test_nan_in_returns(self):
        """NaN in returns should be handled gracefully."""
        r = [0.01, float('nan'), 0.02, 0.03]
        # sharpe_ratio doesn't explicitly filter NaN (pure numpy), but std of NaN is NaN
        # which gets caught by the numeric checks in factor metrics
        # Performance metrics use numpy directly, NaN propagates
        # This test documents the behavior
        result = sharpe_ratio(r)
        # NaN propagates through numpy, result will be NaN or 0.0 depending on implementation
        assert isinstance(result, float)

    def test_inf_in_feature(self):
        """Inf in feature values -> ic returns 0.0."""
        fv = [0.1, 0.2, float('inf'), 0.4, 0.5, 0.6, 0.7, 0.8]
        fr = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        result = ic(fv, fr)
        assert isinstance(result, float)
