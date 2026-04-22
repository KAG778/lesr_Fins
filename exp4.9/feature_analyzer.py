"""
Feature Analyzer for Exp4.9: Regime-Grouped Analysis

Enhancement over 4.7:
- analyze_features now returns regime_importance (per-regime feature scores)
- Falls back gracefully if no regime labels provided
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def analyze_features(
    episode_states: List[np.ndarray],
    episode_rewards: List[float],
    original_dim: int,
    regime_labels: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Analyze feature importance with optional regime grouping.

    Args:
        episode_states: Enhanced states from training
        episode_rewards: Corresponding rewards
        original_dim: Original state dimension (120)
        regime_labels: Regime label for each state (optional)

    Returns:
        importance: Global combined importance
        correlations: Global Spearman correlations
        shap_importance: Global SHAP importance
        regime_importance: Per-regime importance dict
    """
    if len(episode_states) == 0:
        empty = np.array([])
        return empty, empty, empty, {}

    states = np.array(episode_states)
    rewards = np.array(episode_rewards)

    # Extra features start after raw (120) + regime (5) = 125
    feature_start = original_dim + 5  # Skip raw OHLCV + regime vector

    if states.shape[1] > feature_start:
        extra_features = states[:, feature_start:]
    elif states.shape[1] > original_dim:
        extra_features = states[:, original_dim:]
    else:
        logger.warning(f"No extra features found (state dim: {states.shape[1]})")
        extra_features = states

    # Global analysis
    correlations = _compute_correlations(extra_features, rewards)
    shap_importance = _compute_shap_importance(extra_features, rewards)
    importance = 0.5 * correlations + 0.5 * shap_importance

    # Pad to full state dimension
    full_dim = states.shape[1]
    full_importance = np.zeros(full_dim)
    full_correlations = np.zeros(full_dim)
    full_shap = np.zeros(full_dim)

    if states.shape[1] > feature_start:
        full_importance[feature_start:] = importance
        full_correlations[feature_start:] = correlations
        full_shap[feature_start:] = shap_importance
    elif states.shape[1] > original_dim:
        full_importance[original_dim:] = importance
        full_correlations[original_dim:] = correlations
        full_shap[original_dim:] = shap_importance

    # Per-regime analysis
    regime_importance = {}
    if regime_labels and len(regime_labels) == len(episode_states):
        regime_importance = _analyze_by_regime(
            extra_features, rewards, regime_labels
        )

    return full_importance, full_correlations, full_shap, regime_importance


def _compute_correlations(extra_features: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """Spearman correlation for each feature."""
    correlations = []
    for i in range(extra_features.shape[1]):
        if extra_features[:, i].std() == 0:
            correlations.append(0.0)
        else:
            corr, _ = spearmanr(extra_features[:, i], rewards)
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
    return np.array(correlations)


def _compute_shap_importance(extra_features: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """SHAP-based importance via RandomForest."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        import shap

        # Sample for speed
        if extra_features.shape[0] > 5000:
            idx = np.random.choice(extra_features.shape[0], 5000, replace=False)
            features = extra_features[idx]
            target = rewards[idx]
        else:
            features = extra_features
            target = rewards

        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(features, target)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(features)

        if isinstance(shap_values, list):
            return np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            return np.abs(shap_values).mean(axis=0)
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}, using correlation only")
        return _compute_correlations(extra_features, rewards)


def _analyze_by_regime(
    extra_features: np.ndarray,
    rewards: np.ndarray,
    regime_labels: List[str]
) -> Dict[str, np.ndarray]:
    """Compute feature importance per regime."""
    regime_importance = {}

    unique_regimes = set(regime_labels)
    for regime in unique_regimes:
        mask = np.array([r == regime for r in regime_labels])
        if mask.sum() < 10:
            # Not enough data for this regime
            regime_importance[regime] = np.zeros(extra_features.shape[1])
            continue

        regime_features = extra_features[mask]
        regime_rewards = rewards[mask]

        # Use correlation only for per-regime (SHAP too expensive)
        corr = _compute_correlations(regime_features, regime_rewards)
        regime_importance[regime] = corr

    return regime_importance


def rank_features(
    importance: np.ndarray,
    correlations: np.ndarray,
    top_k: int = 5
) -> List[Tuple[int, float, float]]:
    top_indices = np.argsort(importance)[-top_k:][::-1]
    return [(int(idx), float(importance[idx]), float(correlations[idx])) for idx in top_indices]


def generate_feature_summary(
    importance: np.ndarray,
    correlations: np.ndarray,
    original_dim: int,
    regime_importance: Dict[str, np.ndarray] = None
) -> str:
    """Generate human-readable feature summary with regime breakdown."""
    summary = []
    feature_start = original_dim + 5

    if len(importance) > feature_start:
        extra_count = len(importance) - feature_start
        summary.append(f"New Features ({extra_count} total) - Top 5:")
        extra_importance = importance[feature_start:]
        top_extra = np.argsort(extra_importance)[-min(5, len(extra_importance)):][::-1]
        for rank, rel_idx in enumerate(top_extra):
            actual_idx = feature_start + rel_idx
            summary.append(
                f"  new_feature_{rel_idx}: importance={extra_importance[rel_idx]:.3f}, "
                f"corr={correlations[actual_idx]:.3f}"
            )

    if regime_importance:
        summary.append("\nFeature Importance by Regime:")
        for regime, imp in regime_importance.items():
            if imp.max() > 0:
                top_idx = np.argmax(imp)
                summary.append(f"  {regime}: top_feature={top_idx} (importance={imp[top_idx]:.3f})")

    return "\n".join(summary)
