"""
Prompts for Exp4.15: JSON Feature Selection Mode

Complete rewrite from Python code generation to JSON feature selection.
Per CONTEXT.md decisions:
  D-01: Market-aware prompt with ~100 token statistical summary
  D-02: Curated iteration context (~2k tokens, no full history)
  D-03: Feature selection via 4 theme packs
  D-04: JSON output mode (no Python code generation)
  D-10: COT feedback with strategy performance + per-indicator IC/IR
  D-11: Negative guidance with specific rejection reasons
  D-12: Batch feedback across all candidates

Replaces (DELETED):
  - Old INITIAL_PROMPT (167 lines of Python code generation)
  - Old get_financial_cot_prompt() -> new get_cot_feedback()
  - Old get_iteration_prompt() -> new curated version
"""

import re
import json
import numpy as np
from typing import List, Dict, Optional


# ===========================================================================
# INITIAL PROMPT TEMPLATE (D-01, D-03, D-04)
# ===========================================================================

INITIAL_PROMPT_TEMPLATE = """You are a financial quantitative analysis expert selecting trading features for a reinforcement learning strategy.

## Available Indicators (organized by theme)

### Trend Following
- RSI(window: 5-60) - Relative Strength Index, default 14
- MACD(fast: 5-20, slow: 15-60, signal: 3-15) - Moving Average Convergence Divergence
- EMA_Cross(fast: 5-20, slow: 15-60) - EMA Crossover Signal
- Momentum(window: 5-60) - Price Momentum, default 10
- ROC(window: 5-60) - Rate of Change, default 10
- SMA_Cross(fast: 5-20, slow: 15-60) - SMA Crossover Signal
- DEMA(window: 5-60) - Double EMA, default 20
- Williams_Alligator() - Williams Alligator (jaw/teeth/lips)
- TSF(window: 5-30) - Time Series Forecast, default 14

### Volatility
- Bollinger(window: 10-40, num_std: 1.0-3.0) - Bollinger Band Position
- ATR(window: 5-30) - Average True Range, default 14
- Volatility(window: 5-60) - Rolling Volatility, default 20
- Skewness(window: 5-60) - Return Distribution Skewness, default 20
- Kurtosis(window: 5-60) - Return Distribution Kurtosis, default 20

### Mean Reversion
- Stochastic(window: 5-30) - Stochastic Oscillator (%K, %D)
- Williams_R(window: 5-30) - Williams %R
- CCI(window: 5-30) - Commodity Channel Index

### Volume
- OBV() - On-Balance Volume
- Volume_Ratio(window: 5-30) - Volume Ratio, default 20
- ADX(window: 5-30) - Average Directional Index, default 14

## Market Context
{market_stats}

## Output Format (STRICT JSON)
```json
{{
  "features": [
    {{"indicator": "RSI", "params": {{"window": 14}}}},
    {{"indicator": "MACD", "params": {{"fast": 12, "slow": 26, "signal": 9}}}},
    {{"indicator": "Bollinger", "params": {{"window": 20, "num_std": 2.0}}}}
  ],
  "rationale": "Brief explanation of why each indicator was selected based on market context"
}}
```

## Rules
- Select 5-15 indicators (system will filter to best 5-10)
- Diversify across themes (don't pick all from one theme)
- Each indicator must have an economic rationale
- Parameters should be adapted to market context above
- Output ONLY the JSON block, no additional text
"""


# ===========================================================================
# MARKET STATS (D-01: ~100 tokens from training data only)
# ===========================================================================

def get_market_stats(training_states: np.ndarray) -> str:
    """Compute ~100 token statistical summary from training period data.

    CRITICAL: Only uses training period data (leakage prevention per D-01).

    Args:
        training_states: Array of raw training states (each 120d interleaved).

    Returns:
        Formatted string with volatility, trend, volume, return stats.
    """
    if len(training_states) == 0:
        return "No market data available"

    # Extract closing prices from all states
    all_closes = []
    all_volumes = []
    for s in training_states:
        n = len(s) // 6
        for i in range(n):
            all_closes.append(s[i * 6])
            all_volumes.append(s[i * 6 + 4])

    closes = np.array(all_closes, dtype=float)
    volumes = np.array(all_volumes, dtype=float)

    if len(closes) < 2:
        return "Insufficient data"

    # Compute stats
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    volatility = float(np.std(returns)) * 100  # percentage
    avg_return = float(np.mean(returns)) * 100  # percentage

    # Trend: MA(5) vs MA(20) relative difference
    if len(closes) >= 5:
        ma5 = np.mean(closes[-5:])
        ma_all = np.mean(closes)
        trend_val = (ma5 - ma_all) / (ma_all + 1e-10)
        if trend_val > 0.01:
            trend_str = f"+{trend_val:.3f} (uptrend)"
        elif trend_val < -0.01:
            trend_str = f"{trend_val:.3f} (downtrend)"
        else:
            trend_str = f"{trend_val:.3f} (sideways)"
    else:
        trend_str = "N/A"

    avg_vol = float(np.mean(volumes))

    return (f"Volatility: {volatility:.1f}% daily | "
            f"Trend: {trend_str} | "
            f"Avg Volume: {avg_vol:.0f} | "
            f"Avg Return: {avg_return:+.3f}%")


# ===========================================================================
# RENDER INITIAL PROMPT
# ===========================================================================

def render_initial_prompt(training_states: np.ndarray) -> str:
    """Render initial prompt with market stats injected.

    Args:
        training_states: Array of raw training states.

    Returns:
        Complete initial prompt string ready for LLM.
    """
    market_stats = get_market_stats(training_states)
    return INITIAL_PROMPT_TEMPLATE.format(market_stats=market_stats)


# ===========================================================================
# ITERATION PROMPT (D-02: curated ~2k tokens)
# ===========================================================================

def get_iteration_prompt(last_selection: Dict, cot_feedback: str,
                         best_selection: Dict, best_score: Dict) -> str:
    """Generate curated iteration prompt (~2k tokens per D-02).

    Only includes last round selection + feedback + best historical, NOT full
    iteration history.

    Args:
        last_selection: Dict with 'features' list from last iteration.
        cot_feedback: COT feedback text from last iteration.
        best_selection: Dict with best historical selection.
        best_score: Dict with best Sharpe and other metrics.

    Returns:
        Curated iteration prompt string.
    """
    last_json = json.dumps(last_selection.get('features', []), indent=2)
    best_json = json.dumps(best_selection.get('features', []), indent=2)
    best_sharpe = best_score.get('sharpe', 0.0)

    prompt = f"""You are a financial quantitative analysis expert improving trading feature selections.

## Last Iteration Selection
```json
{last_json}
```

## Feedback from Last Iteration
{cot_feedback}

## Best Historical Selection (Sharpe: {best_sharpe:.3f})
```json
{best_json}
```

## Available Indicators (abbreviated)
Trend: RSI(w), MACD(f,s,sig), EMA_Cross(f,s), Momentum(w), ROC(w), SMA_Cross(f,s), DEMA(w), Williams_Alligator(), TSF(w)
Volatility: Bollinger(w,std), ATR(w), Volatility(w), Skewness(w), Kurtosis(w)
Mean Reversion: Stochastic(w), Williams_R(w), CCI(w)
Volume: OBV(), Volume_Ratio(w), ADX(w)

## Output Format (STRICT JSON)
```json
{{{{
  "features": [{{{{"indicator": "...", "params": {{{{...}}}}}}}}],
  "rationale": "..."
}}}}
```

## Instructions
- Select 5-15 indicators based on feedback above
- Avoid indicators that were rejected in previous iterations
- Diversify across themes
- Parameters within listed ranges
- Output ONLY JSON

"""
    return prompt[:3000]  # Hard cap at ~3000 chars (~2k tokens)


# ===========================================================================
# JSON EXTRACTION
# ===========================================================================

def _extract_json(text: str) -> dict:
    """Extract and parse JSON from LLM output text.

    Tries markdown code block extraction first, then falls back to finding
    first { to last }. Handles trailing commas and type conversion.

    Args:
        text: Raw LLM output text.

    Returns:
        Parsed dict with 'features' key.

    Raises:
        ValueError: If no valid JSON could be extracted.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Empty or non-string input")

    # Try markdown code block extraction first
    md_pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
    md_match = re.search(md_pattern, text, re.DOTALL)
    json_str = md_match.group(1).strip() if md_match else None

    if json_str is None:
        # Fall back to finding first { to last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace == -1 or last_brace == -1 or first_brace > last_brace:
            raise ValueError("No JSON object found in text")
        json_str = text[first_brace:last_brace + 1]

    # Clean up trailing commas (JSON5 -> JSON)
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}")

    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object (dict), got {type(result).__name__}")

    # Convert string params to int/float where appropriate
    if 'features' in result and isinstance(result['features'], list):
        for feature in result['features']:
            if isinstance(feature, dict) and 'params' in feature:
                for key, val in feature['params'].items():
                    if isinstance(val, str):
                        try:
                            feature['params'][key] = int(val)
                        except ValueError:
                            try:
                                feature['params'][key] = float(val)
                            except ValueError:
                                pass  # keep as string

    return result


# ===========================================================================
# COT FEEDBACK (D-10, D-11, D-12)
# ===========================================================================

def get_cot_feedback(selections: List[Dict], scores: List[Dict],
                     screening_reports: List[Dict],
                     stability_reports: List[Dict]) -> str:
    """Generate structured COT feedback replacing old get_financial_cot_prompt.

    Per D-10: Performance (Sharpe/MaxDD/TotalReturn) + per-indicator IC/IR
    Per D-11: Negative guidance with specific rejection reasons
    Per D-12: Batch feedback across all candidates

    Args:
        selections: List of candidate selections (each with 'features' key).
        scores: List of performance score dicts (sharpe, max_dd, total_return).
        screening_reports: List of screening reports per candidate.
        stability_reports: List of stability reports per candidate.

    Returns:
        Structured COT feedback string.
    """
    n = len(selections)
    lines = []
    lines.append(f"Training results for {n} candidate feature selections:\n")

    best_idx = 0
    best_sharpe = -float('inf')
    all_rejected = []

    for i in range(n):
        lines.append(f"========== Candidate {i + 1} ==========")

        # Selection
        features = selections[i].get('features', [])
        feature_names = [f.get('indicator', '?') for f in features]
        lines.append(f"Selected indicators: {', '.join(feature_names)}")

        # Performance (D-10)
        if i < len(scores):
            s = scores[i]
            lines.append(f"Performance:")
            lines.append(f"  Sharpe Ratio: {s.get('sharpe', 0.0):.3f}")
            lines.append(f"  Max Drawdown: {s.get('max_dd', 0.0):.2f}%")
            lines.append(f"  Total Return: {s.get('total_return', 0.0):.2f}%")
            if s.get('sharpe', 0) > best_sharpe:
                best_sharpe = s.get('sharpe', 0)
                best_idx = i

        # Per-indicator IC/IR from screening (D-10)
        if i < len(screening_reports):
            report = screening_reports[i]
            metrics = report.get('feature_metrics', {})
            if metrics:
                lines.append("Per-indicator metrics:")
                for name, m in metrics.items():
                    ic_val = m.get('ic', 0.0)
                    ir_val = m.get('ir', m.get('variance', 0.0))
                    lines.append(f"  {name}: IC={ic_val:.4f}, Variance={ir_val:.6f}")

            # Rejected indicators (D-11)
            rejected = report.get('rejected', [])
            if rejected:
                lines.append("Rejected indicators:")
                for r in rejected:
                    reason = r.get('reason', 'unknown')
                    all_rejected.append(r)
                    lines.append(f"  {r.get('indicator', '?')}: {reason}")

        # Stability info
        if i < len(stability_reports):
            stab = stability_reports[i]
            unstable = stab.get('unstable_features', [])
            if unstable:
                lines.append("Unstable features:")
                for u in unstable:
                    lines.append(f"  {u.get('indicator', '?')}: {u.get('reason', 'unstable')}")

        lines.append("")

    # Summary section (D-11, D-12)
    lines.append("========== Summary ==========")
    lines.append(f"Best candidate: #{best_idx + 1} (Sharpe: {best_sharpe:.3f})")
    lines.append(f"Candidate comparison: Sharpe ranges from "
                 f"{min(s.get('sharpe', 0) for s in scores):.3f} to "
                 f"{max(s.get('sharpe', 0) for s in scores):.3f}")

    # Negative guidance (D-11)
    if all_rejected:
        lines.append("")
        lines.append("Negative guidance (avoid these patterns):")
        seen_reasons = set()
        for r in all_rejected:
            indicator = r.get('indicator', '?')
            reason = r.get('reason', '')
            key = f"{indicator}:{reason}"
            if key not in seen_reasons:
                seen_reasons.add(key)
                lines.append(f"  - Avoid {indicator}: {reason}")

    lines.append("")
    lines.append("Do not select indicators with IC below 0.02 threshold.")
    lines.append("Avoid selecting multiple same-type indicators with similar parameters.")
    lines.append("Consider diversifying across themes for better stability.")

    return "\n".join(lines)


# ===========================================================================
# BACKWARD COMPATIBILITY ALIASES
# ===========================================================================
# lesr_controller.py still imports these old names. They will be updated in
# Plan 03-03 when the controller is rewritten. Until then, provide aliases.

# Old INITIAL_PROMPT alias (was a 167-line Python code-gen prompt)
INITIAL_PROMPT = INITIAL_PROMPT_TEMPLATE


def get_financial_cot_prompt(codes, scores, importance, correlations,
                             original_dim, worst_trades=None):
    """Backward-compatible wrapper: old signature -> new get_cot_feedback.

    Will be removed when lesr_controller.py is updated in Plan 03-03.
    """
    # Convert old signature to new-style inputs
    selections = []
    for code in codes:
        selections.append({"features": [], "code": code})

    screening_reports = [{"screened_selection": [], "feature_metrics": {},
                          "rejected": []} for _ in scores]
    stability_reports = [{"stability_report": {}, "unstable_features": []}
                         for _ in scores]

    return get_cot_feedback(selections, scores, screening_reports,
                            stability_reports)
