"""
IC (Information Coefficient) Analyzer for LESR COT Feedback

Computes per-dimension IC between revised state features and forward returns.
Generates four-tier COT feedback:
  1. Market environment context
  2. Strong features (|IC| > strong_threshold)
  3. Weak features (|IC| < weak_threshold)
  4. Negative IC features (IC < -0.03)
  5. Missing analysis suggestions
  6. Intrinsic reward diagnosis
"""

import numpy as np
from typing import Dict, List, Optional


def compute_ic_profile(revised_states: np.ndarray,
                       forward_returns: np.ndarray) -> Dict[int, float]:
    """Compute Pearson IC between each extra state dim and forward returns.

    Args:
        revised_states: (N, state_dim) array of revised states.
        forward_returns: (N,) array of forward portfolio returns.

    Returns:
        Dict mapping extra dim index to IC value.
    """
    if revised_states.ndim != 2 or forward_returns.ndim != 1:
        return {}
    if revised_states.shape[0] != forward_returns.shape[0]:
        return {}
    if revised_states.shape[0] < 10:
        return {}

    n_dims = revised_states.shape[1]
    extra_start = 120
    if n_dims <= extra_start:
        return {}

    ic_profile = {}
    for dim in range(extra_start, n_dims):
        col = revised_states[:, dim]
        if np.std(col) < 1e-10:
            ic_profile[dim] = 0.0
            continue
        ret_std = np.std(forward_returns)
        if ret_std < 1e-10:
            ic_profile[dim] = 0.0
            continue
        corr = np.corrcoef(col, forward_returns)[0, 1]
        ic_profile[dim] = float(corr) if not np.isnan(corr) else 0.0

    return ic_profile


def compute_regime_specific_ic(revised_states: np.ndarray,
                               forward_returns: np.ndarray,
                               regime_labels: np.ndarray) -> Dict[str, Dict[int, float]]:
    """Compute IC per market regime.

    Args:
        revised_states: (N, state_dim) array.
        forward_returns: (N,) array.
        regime_labels: (N,) array of regime strings: 'trending_up', 'volatile', 'trending_down'.

    Returns:
        Dict mapping regime name to {dim: ic_value}.
    """
    result = {}
    for regime in set(regime_labels):
        mask = regime_labels == regime
        if mask.sum() < 10:
            continue
        result[regime] = compute_ic_profile(revised_states[mask], forward_returns[mask])
    return result


def _classify_regime(trend: float, vol: float) -> str:
    """Classify a single timestep's regime from trend/vol values."""
    if vol > 0.6:
        return 'volatile'
    elif trend > 0.3:
        return 'trending_up'
    elif trend < -0.3:
        return 'trending_down'
    else:
        return 'neutral'


def build_ic_cot_prompt(
    sample_results: List[Dict],
    best_idx: int,
    strong_threshold: float = 0.05,
    weak_threshold: float = 0.02,
    market_period_summary: str = "",
) -> str:
    """Build IC-based COT feedback prompt for LLM.

    Args:
        sample_results: list of dicts, each with keys:
            'code': str
            'performance': dict with 'sharpe', 'total_return', 'max_drawdown'
            'ic_profile': dict[int, float] from compute_ic_profile
            'regime_ic': dict[str, dict[int, float]] from compute_regime_specific_ic
            'intrinsic_reward_stats': dict with 'mean', 'correlation_with_performance'
        best_idx: index of the best-performing sample
        strong_threshold: IC above this = strong feature
        weak_threshold: IC below this = weak feature
        market_period_summary: text describing training period market conditions

    Returns:
        Formatted COT feedback string.
    """
    lines = []

    for i, sample in enumerate(sample_results):
        code = sample.get('code', '')
        perf = sample.get('performance', {})
        ic_profile = sample.get('ic_profile', {})
        regime_ic = sample.get('regime_ic', {})
        ir_stats = sample.get('intrinsic_reward_stats', {})

        marker = " (BEST)" if i == best_idx else ""
        lines.append(f"========== Code Sample {i+1}{marker} "
                     f"(Sharpe={perf.get('sharpe', 0):.3f}, "
                     f"Return={perf.get('total_return', 0):.2f}%, "
                     f"MaxDD={perf.get('max_drawdown', 0):.2f}%) ==========")
        lines.append(code)
        lines.append("")

        if not ic_profile:
            lines.append("  (No IC profile computed - insufficient data)")
            lines.append("")
            continue

        lines.append("  [IC Analysis]")
        strong, weak, negative = [], [], []
        for dim, ic_val in sorted(ic_profile.items()):
            abs_ic = abs(ic_val)
            tag = ""
            if abs_ic > strong_threshold:
                tag = "<- Strong"
                strong.append((dim, ic_val))
            elif abs_ic < weak_threshold:
                tag = "<- Weak"
                weak.append((dim, ic_val))
            if ic_val < -0.03:
                tag = "<- Negative"
                negative.append((dim, ic_val))
            lines.append(f"    s[{dim}]: IC = {ic_val:+.4f} {tag}")

        if regime_ic:
            lines.append("  [Regime-Specific IC]")
            for regime, dim_ics in regime_ic.items():
                lines.append(f"    {regime}:")
                for dim, ic_val in sorted(dim_ics.items()):
                    lines.append(f"      s[{dim}]: IC = {ic_val:+.4f}")

        if ir_stats:
            lines.append("  [Intrinsic Reward Diagnosis]")
            lines.append(f"    Mean intrinsic_reward = {ir_stats.get('mean', 0):.6f}")
            lines.append(f"    Correlation with performance = "
                         f"{ir_stats.get('correlation_with_performance', 0):.3f}")
            corr = ir_stats.get('correlation_with_performance', 0)
            if abs(corr) > 0.3:
                lines.append(f"    -> Strong guidance effect")
            elif abs(corr) > 0.1:
                lines.append(f"    -> Moderate guidance effect")
            else:
                lines.append(f"    -> Weak guidance effect, consider redesigning")

        lines.append("")

    lines.append("[Improvement Suggestions]")
    best_sample = sample_results[best_idx] if best_idx < len(sample_results) else {}
    best_ic = best_sample.get('ic_profile', {})
    best_regime_ic = best_sample.get('regime_ic', {})

    if best_ic:
        dims_sorted = sorted(best_ic.items(), key=lambda x: abs(x[1]), reverse=True)
        if dims_sorted:
            best_dim, best_ic_val = dims_sorted[0]
            lines.append(f"  (a) Strongest feature: s[{best_dim}] (IC={best_ic_val:+.4f}). "
                         "Consider building derivatives or multi-horizon versions.")

        weak_dims = [(d, v) for d, v in best_ic.items() if abs(v) < weak_threshold]
        if weak_dims:
            dim_str = ", ".join(f"s[{d}]" for d, _ in weak_dims)
            lines.append(f"  (b) Weak features: {dim_str}. Consider replacing with "
                         "more informative signals (volatility-adjusted momentum, "
                         "cross-sectional rank, etc.).")

        neg_dims = [(d, v) for d, v in best_ic.items() if v < -0.03]
        if neg_dims:
            dim_str = ", ".join(f"s[{d}] (IC={v:+.4f})" for d, v in neg_dims)
            lines.append(f"  (c) Negative IC features: {dim_str}. These may be harmful. "
                         "Consider removing or inverting the signal.")

        has_volatility = any('volatil' in str(v) for v in best_ic.values())
        if not has_volatility and best_regime_ic.get('volatile'):
            lines.append("  (d) No volatility features detected for volatile regime. "
                         "Consider adding realized_volatility or downside_risk.")

    lines.append("")
    return "\n".join(lines)
