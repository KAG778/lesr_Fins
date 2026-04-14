"""
Tests for compute_feature_quality (DIAG-03).

Validates per-feature metrics, degenerate detection, correlation correctness,
NaN handling, and aggregate calculations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# exp4.7 directory contains a dot, so standard Python imports fail.
# Add exp4.7 to sys.path so we can import diagnosis sub-package directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagnosis.feature_quality import compute_feature_quality


class TestComputesPerFeatureMetrics:
    """test_computes_per_feature_metrics"""

    def test_computes_per_feature_metrics(self, synthetic_states, synthetic_rewards):
        result = compute_feature_quality(synthetic_states, synthetic_rewards, original_dim=120)

        assert "per_feature" in result
        assert "aggregate" in result
        per = result["per_feature"]
        # 130 - 120 = 10 extra features
        assert len(per) == 10

        for entry in per:
            for key in ("index", "variance", "correlation", "p_value", "information_ratio"):
                assert key in entry, f"Missing key: {key}"
            # No NaN values
            for key in ("variance", "correlation", "p_value", "information_ratio"):
                assert not np.isnan(entry[key]), f"NaN found in {key} for feature {entry['index']}"


class TestDetectsDegenerateFeatures:
    """test_detects_degenerate_features"""

    def test_detects_degenerate_features(self, degenerate_states):
        np.random.seed(42)
        rewards = np.random.randn(50)
        result = compute_feature_quality(degenerate_states, rewards, original_dim=120)

        assert result["aggregate"]["num_degenerate"] == 5
        # Each degenerate feature should have correlation == 0.0
        for feat in result["per_feature"]:
            assert feat["correlation"] == 0.0
            assert feat["p_value"] == 1.0


class TestCorrelationsArePositiveForCorrelated:
    """test_correlations_are_positive_for_correlated"""

    def test_correlations_are_positive_for_correlated(self):
        np.random.seed(42)
        # Build states where extra feature 0 is perfectly correlated with rewards
        n = 200
        base = np.random.randn(n, 120)
        rewards = np.random.randn(n)
        extra = np.zeros((n, 1))
        extra[:, 0] = rewards  # Perfect correlation
        states = np.hstack([base, extra])

        result = compute_feature_quality(states, rewards, original_dim=120)
        assert result["per_feature"][0]["correlation"] > 0.9


class TestHandlesZeroStd:
    """test_handles_zero_std"""

    def test_handles_zero_std(self):
        np.random.seed(42)
        n = 100
        base = np.random.randn(n, 120)
        extra = np.ones((n, 1)) * 42.0  # Constant feature
        states = np.hstack([base, extra])
        rewards = np.random.randn(n)

        result = compute_feature_quality(states, rewards, original_dim=120)
        feat = result["per_feature"][0]

        assert feat["correlation"] == 0.0
        assert not np.isnan(feat["variance"])
        assert not np.isnan(feat["correlation"])
        assert not np.isnan(feat["information_ratio"])


class TestAggregateMetricsCorrect:
    """test_aggregate_metrics_correct"""

    def test_aggregate_metrics_correct(self, synthetic_states, synthetic_rewards):
        result = compute_feature_quality(synthetic_states, synthetic_rewards, original_dim=120)

        per = result["per_feature"]
        corrs = [f["correlation"] for f in per]
        p_vals = [f["p_value"] for f in per]

        # mean_abs_correlation
        assert abs(result["aggregate"]["mean_abs_correlation"] - np.mean(corrs)) < 1e-10
        # max_abs_correlation
        assert abs(result["aggregate"]["max_abs_correlation"] - max(corrs)) < 1e-10
        # num_significant: count features with p < 0.05
        expected_significant = sum(1 for p in p_vals if p < 0.05)
        assert result["aggregate"]["num_significant"] == expected_significant


class TestHandlesNanRewards:
    """test_handles_nan_rewards"""

    def test_handles_nan_rewards(self, synthetic_states):
        np.random.seed(42)
        rewards = np.random.randn(200)
        # Sprinkle NaN values
        rewards[10] = np.nan
        rewards[50] = np.nan
        rewards[150] = np.nan

        # Should not crash
        result = compute_feature_quality(synthetic_states, rewards, original_dim=120)

        # Verify no NaN in output
        for feat in result["per_feature"]:
            assert not np.isnan(feat["variance"])
            assert not np.isnan(feat["correlation"])
            assert not np.isnan(feat["p_value"])
            assert not np.isnan(feat["information_ratio"])

        for val in result["aggregate"].values():
            assert not np.isnan(val)
