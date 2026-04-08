"""
Feature Analyzer Module for Exp4.7 Financial Trading Experiment

This module implements feature importance analysis using correlation and SHAP values.
It replaces the Lipschitz continuity analysis used in the original LESR framework,
as financial data violates the Lipschitz assumptions (continuity, smoothness, low noise).
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
import shap
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def analyze_features(
    episode_states: List[np.ndarray],
    episode_rewards: List[float],
    original_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze feature importance using correlation and SHAP.

    This replaces Lipschitz analysis because financial data:
    - Has discontinuous jumps (price gaps, limit hits)
    - Has high noise (SNR ≈ 8dB)
    - Is non-stationary (distribution drift over time)

    Args:
        episode_states: List of enhanced states from training episodes
        episode_rewards: List of corresponding rewards
        original_dim: Original state dimension (120 for OHLCV data)

    Returns:
        importance: Combined importance score (0.5 * correlation + 0.5 * SHAP)
        correlations: Absolute Spearman correlation for each feature
        shap_importance: SHAP importance for each feature
    """
    if len(episode_states) == 0:
        logger.warning("No episode states provided for analysis")
        return np.array([]), np.array([]), np.array([])

    states = np.array(episode_states)
    rewards = np.array(episode_rewards)

    # Only analyze extra features (beyond original 120 dimensions)
    if states.shape[1] > original_dim:
        extra_features = states[:, original_dim:]
    else:
        logger.warning(f"No extra features found (state dim: {states.shape[1]}, original: {original_dim})")
        # Analyze all features if no extra features
        extra_features = states

    # Method 1: Spearman correlation (robust to outliers)
    correlations = []
    for i in range(extra_features.shape[1]):
        if extra_features[:, i].std() == 0:
            # Constant feature, zero correlation
            correlations.append(0.0)
        else:
            corr, p_value = spearmanr(extra_features[:, i], rewards)
            if np.isnan(corr):
                corr = 0.0
            correlations.append(abs(corr))  # Take absolute value

    correlations = np.array(correlations)

    # Method 2: SHAP analysis (captures non-linear relationships)
    try:
        logger.info(f"开始SHAP分析，数据形状: {extra_features.shape}")
        print(f"开始SHAP分析，数据形状: {extra_features.shape}", flush=True)

        # 采样加速：如果样本数超过5000，随机采样5000个
        if extra_features.shape[0] > 5000:
            sample_idx = np.random.choice(extra_features.shape[0], 5000, replace=False)
            shap_features = extra_features[sample_idx]
            shap_rewards = rewards[sample_idx]
            logger.info(f"SHAP采样: 使用5000个样本代替{extra_features.shape[0]}个")
            print(f"SHAP采样: 使用5000个样本代替{extra_features.shape[0]}个", flush=True)
        else:
            shap_features = extra_features
            shap_rewards = rewards

        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        logger.info("训练RandomForest...")
        print("训练RandomForest...", flush=True)
        rf.fit(shap_features, shap_rewards)
        logger.info("RandomForest训练完成")
        print("RandomForest训练完成", flush=True)

        logger.info("计算SHAP值...")
        print("计算SHAP值...", flush=True)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(shap_features)
        logger.info("SHAP值计算完成")
        print("SHAP值计算完成", flush=True)

        if isinstance(shap_values, list):
            # For multi-output, take mean across outputs
            shap_values = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}, using correlation only")
        shap_importance = correlations.copy()

    # Combined importance: weighted average
    importance = 0.5 * correlations + 0.5 * shap_importance

    # Pad to full state dimension if needed
    full_importance = np.zeros(states.shape[1])
    full_correlations = np.zeros(states.shape[1])
    full_shap = np.zeros(states.shape[1])

    if states.shape[1] > original_dim:
        # Original features get zero importance (focus on new features)
        full_importance[original_dim:] = importance
        full_correlations[original_dim:] = correlations
        full_shap[original_dim:] = shap_importance
    else:
        full_importance = importance
        full_correlations = correlations
        full_shap = shap_importance

    return full_importance, full_correlations, full_shap


def rank_features(
    importance: np.ndarray,
    correlations: np.ndarray,
    top_k: int = 5
) -> List[Tuple[int, float, float]]:
    """
    Rank features by importance and return top-k.

    Args:
        importance: Feature importance scores
        correlations: Feature correlation scores
        top_k: Number of top features to return

    Returns:
        List of (feature_index, importance, correlation) tuples
    """
    top_indices = np.argsort(importance)[-top_k:][::-1]

    ranked = [
        (int(idx), float(importance[idx]), float(correlations[idx]))
        for idx in top_indices
    ]

    return ranked


def generate_feature_summary(
    importance: np.ndarray,
    correlations: np.ndarray,
    original_dim: int
) -> str:
    """
    Generate a human-readable summary of feature analysis.

    Args:
        importance: Feature importance scores
        correlations: Feature correlation scores
        original_dim: Original state dimension

    Returns:
        Formatted string summarizing feature analysis
    """
    summary = []

    # Original features (OHLCV)
    summary.append("Original Features (OHLCV) - Top 5:")
    original_importance = importance[:original_dim]
    top_original = np.argsort(original_importance)[-5:][::-1]
    for idx in top_original:
        if original_importance[idx] > 0:
            summary.append(f"  s[{idx}]: importance={original_importance[idx]:.3f}, corr={correlations[idx]:.3f}")

    # Extra features (LLM-generated)
    if len(importance) > original_dim:
        extra_count = len(importance) - original_dim
        summary.append(f"\nNew Features ({extra_count} total) - Top 5:")
        extra_importance = importance[original_dim:]
        top_extra = np.argsort(extra_importance)[-min(5, len(extra_importance)):][::-1]
        for rank, rel_idx in enumerate(top_extra):
            actual_idx = original_dim + rel_idx
            summary.append(
                f"  new_feature_{rel_idx}: importance={extra_importance[rel_idx]:.3f}, "
                f"corr={correlations[actual_idx]:.3f}"
            )

    return "\n".join(summary)
