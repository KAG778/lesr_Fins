"""
Feature Analyzer for Exp4.9_c

Simplified from 4.9: just global analysis, no regime grouping.
Regime is handled in the trainer/controller layers.
"""
import numpy as np
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


def analyze_features(states, rewards, original_dim):
    """
    Analyze feature importance.
    
    Args:
        states: array of enhanced states
        rewards: array of rewards
        original_dim: 120 (raw state)
    
    Returns:
        importance, correlations, shap_importance (all padded to full state dim)
    """
    if len(states) == 0:
        d = states.shape[1] if hasattr(states, 'shape') else 0
        z = np.zeros(max(d, 1))
        return z, z, z

    states = np.array(states)
    rewards = np.array(rewards)
    full_dim = states.shape[1]

    # Analyze features beyond raw+regime (123+)
    feature_start = original_dim + 3  # 120 raw + 3 regime
    if full_dim > feature_start:
        extra = states[:, feature_start:]
    else:
        extra = states

    correlations = _spearman(extra, rewards)
    shap = _shap_importance(extra, rewards)
    combined = 0.5 * correlations + 0.5 * shap

    # Pad to full dim
    full_corr = np.zeros(full_dim)
    full_shap = np.zeros(full_dim)
    full_imp = np.zeros(full_dim)
    if full_dim > feature_start:
        full_corr[feature_start:] = correlations
        full_shap[feature_start:] = shap
        full_imp[feature_start:] = combined

    return full_imp, full_corr, full_shap


def _spearman(features, rewards):
    corrs = []
    for i in range(features.shape[1]):
        if features[:, i].std() < 1e-10:
            corrs.append(0.0)
        else:
            c, _ = spearmanr(features[:, i], rewards)
            corrs.append(abs(c) if not np.isnan(c) else 0.0)
    return np.array(corrs)


def _shap_importance(features, rewards):
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(features, rewards)
        return np.array([max(0, x) for x in rf.feature_importances_])
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        return _spearman(features, rewards)
