"""
Feature Quality Module for LESR Diagnosis Framework (DIAG-03).

Computes per-sample feature quality diagnostics (variance, return correlation,
information ratio) for LLM-generated features.  Degenerate features (zero
variance) are detected and flagged without producing NaN.
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List


def compute_feature_quality(
    states: np.ndarray,
    rewards: np.ndarray,
    original_dim: int = 123,
) -> dict:
    """Compute quality metrics for LLM-generated features.

    Analyses features beyond *original_dim* in the state array.

    Args:
        states: 2-D array of shape ``(N, D)`` where ``D >= original_dim``.
        rewards: 1-D array of shape ``(N,)`` with episode rewards.
        original_dim: Index separating original features from LLM-generated
            extra features (default 123: 120 OHLCV + 3 regime).

    Returns:
        Dict with:
          - ``per_feature``: list of dicts, each with keys *index*,
            *variance*, *correlation*, *p_value*, *information_ratio*.
          - ``aggregate``: dict with keys *mean_abs_correlation*,
            *max_abs_correlation*, *num_degenerate*, *num_significant*.
    """
    extra_features = states[:, original_dim:]
    num_extra = extra_features.shape[1]

    # Guard NaN in rewards: replace NaN with 0 for computation
    clean_rewards = rewards.copy()
    nan_mask = np.isnan(clean_rewards)
    if nan_mask.any():
        clean_rewards[nan_mask] = 0.0

    per_feature: List[Dict] = []
    abs_correlations: List[float] = []
    num_degenerate = 0
    num_significant = 0

    for i in range(num_extra):
        feat = extra_features[:, i]
        variance = float(np.var(feat))

        std = float(np.std(feat))
        if std > 1e-10:
            corr_result = spearmanr(feat, clean_rewards)
            corr = float(corr_result.correlation) if not np.isnan(corr_result.correlation) else 0.0
            p_val = float(corr_result.pvalue) if not np.isnan(corr_result.pvalue) else 1.0

            # Information ratio: mean(aligned_returns) / std(aligned_returns)
            aligned_returns = clean_rewards * np.sign(corr)
            aligned_std = float(np.std(aligned_returns))
            if aligned_std > 1e-10:
                info_ratio = float(np.mean(aligned_returns)) / aligned_std
            else:
                info_ratio = 0.0
        else:
            # Degenerate feature: zero or near-zero variance
            corr = 0.0
            p_val = 1.0
            info_ratio = 0.0
            num_degenerate += 1

        abs_corr = abs(corr)
        abs_correlations.append(abs_corr)
        if p_val < 0.05:
            num_significant += 1

        per_feature.append({
            "index": i,
            "variance": variance,
            "correlation": abs_corr,
            "p_value": p_val,
            "information_ratio": info_ratio,
        })

    aggregate = {
        "mean_abs_correlation": float(np.mean(abs_correlations)) if abs_correlations else 0.0,
        "max_abs_correlation": float(max(abs_correlations)) if abs_correlations else 0.0,
        "num_degenerate": num_degenerate,
        "num_significant": num_significant,
    }

    return {
        "per_feature": per_feature,
        "aggregate": aggregate,
    }
