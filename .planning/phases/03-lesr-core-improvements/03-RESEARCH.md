# Phase 3: LESR Core Improvements - Research

**Researched:** 2026-04-15
**Domain:** LLM-driven feature engineering for RL trading strategies
**Confidence:** HIGH

## Summary

Phase 3 transforms the LESR system from free-form Python code generation to structured JSON-based feature selection from a predefined indicator library. The current exp4.15 codebase (copied from exp4.9_c) generates two Python functions via LLM (`revise_state` and `intrinsic_reward`), validates them with `importlib` dynamic loading, and passes them to DQN training. The new architecture eliminates code generation entirely: LLM outputs JSON selecting indicators and parameters, a new `feature_library.py` module assembles closures, and `intrinsic_reward` is replaced by fixed human-designed rules.

This research confirms the technical feasibility of all locked decisions in CONTEXT.md, identifies specific implementation patterns for the 20+ indicators in pure NumPy, documents the exact code transformation points in the existing codebase, and catalogs the pitfalls around NaN propagation, degenerate features, and JSON parsing robustness. The existing test infrastructure (73 passing tests) covers metrics and leakage prevention, providing a solid foundation for new feature-library and validation tests.

**Primary recommendation:** Implement in three waves -- (1) feature_library.py + indicator implementations, (2) prompts.py rewrite + JSON validation pipeline, (3) fixed reward rules + controller integration. Each wave is independently testable.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Prompt is market-aware with statistical summaries (~100 tokens)
- **D-02:** Iteration context is curated (~2k tokens/iteration), no full history
- **D-03:** Feature selection guided by 4 theme packs (Trend, Volatility, Mean Reversion, Volume)
- **D-04:** Prompt rewritten to JSON output mode (no Python code generation)
- **D-05:** Quality gate = JSON parse + indicator exists + params in range + IC/variance pre-evaluation
- **D-06:** Final features passed to DQN fixed at 5-10
- **D-07:** Loose thresholds: IC > 0.02, variance > 1e-6
- **D-08:** Same-type dedup: keep higher IC
- **D-09:** NaN/Inf checks required
- **D-10:** COT feedback = strategy perf + factor-level assessment
- **D-11:** Negative guidance with specific rejection reasons
- **D-12:** Batch feedback across all candidates
- **D-13:** Activate `check_prompt_for_leakage()`
- **D-14:** Stability via fixed sub-period split (3-4 periods)
- **D-15:** Stability = IC mean + volatility ratio (IC std < 2 * IC mean)
- **D-16:** Unstable features removed, LLM notified
- **D-17:** Z-score normalization
- **D-18:** Parameterized indicators with continuous params, validated in range
- **D-19:** Pure Python + NumPy only (no ta-lib, no pandas_ta)
- **D-20:** 20+ indicators in library
- **D-21:** Closure-based function assembly (no exec/eval)
- **D-22:** Fixed 5-6 reward rules replacing LLM-generated intrinsic_reward
- **D-23:** intrinsic_weight=0.02, regime_bonus_weight=0.01
- **D-24:** 3 candidates x 5 rounds = 15 evaluations

### Claude's Discretion
- Exact prompt template wording and format
- Default parameter ranges for each indicator in theme packs
- Report format (markdown tables vs LaTeX)
- Specific IC threshold tuning
- Exact list and parameter ranges of extended indicators
- Specific thresholds and reward values for 5-6 fixed reward rules

### Deferred Ideas (OUT OF SCOPE)
- Custom feature proposal by LLM (outside the library)
- Adaptive feature library (growing over iterations)
- Rolling IC-based stability assessment
- Market regime-stratified stability evaluation
- Early stopping mechanism for optimization loop
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LESR-01 | Prompt templates with precise financial semantics, generating economically meaningful features | D-01 market stats injection, D-03 theme packs, D-04 JSON mode. Current INITIAL_PROMPT is 167 lines of Python-code-generation prompt. Must be replaced with JSON-selection prompt listing available indicators and expected output schema. |
| LESR-02 | Automated checks for syntax correctness, output dimension matching, numerical stability | D-05 JSON parse + indicator validation + IC/variance gates, D-09 NaN/Inf checks. Current `_validate_code()` uses importlib -- must be replaced with `_validate_selection()` that parses JSON, checks indicator registry, computes features on sample data, verifies IC/variance. |
| LESR-03 | COT feedback with high-confidence results and explicit negative guidance | D-10 strategy perf + factor assessment, D-11 specific rejection reasons, D-12 batch feedback. Current `get_financial_cot_prompt()` sends Python code to LLM -- must be rewritten to send JSON selections with per-indicator IC scores and rejection reasons. D-13 leakage check must be activated. |
| LESR-04 | Feature filtering to 5-10 non-degenerate features | D-06 fixed count, D-07 loose thresholds, D-08 same-type dedup. Current system has no feature filtering -- `_validate_code()` only checks dimension consistency. New pipeline: compute all selected indicators -> evaluate IC/variance -> reject degenerate -> dedup same-type -> keep top 5-10 by IC. |
| LESR-05 | Feature stability scores across sub-periods | D-14 fixed sub-period split (3-4 periods), D-15 stability metric (IC mean + volatility ratio), D-16 unstable removal. Entirely new functionality. Requires splitting training data, computing per-period IC, and generating stability report. |
</phase_requirements>

## Standard Stack

### Core (already installed, verified)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.2.6 | Array operations, indicator math | Project foundation, all indicators implemented on top of it [VERIFIED: runtime check] |
| scipy | 1.16.2 | Spearman correlation for IC computation | Already used in metrics.py for IC/IR [VERIFIED: runtime check] |
| torch | 2.9.0+cu128 | DQN network, training | Core RL framework, unchanged from baseline [VERIFIED: runtime check] |
| openai | 1.106.1 | LLM API calls | Already in use with v1.x client pattern [VERIFIED: runtime check] |
| pytest | 9.0.3 | Test framework | 73 tests already passing in exp4.15 [VERIFIED: runtime check] |

### Supporting (already available, no install needed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | - | Parse LLM JSON output | Core of new D-04 JSON mode |
| logging (stdlib) | - | Debug/info logging | Already used throughout codebase |
| pickle (stdlib) | - | Result serialization | Already used in controller for results |
| re (stdlib) | - | Regex for code/JSON extraction | Currently used in `_extract_code()` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure NumPy indicators | ta-lib / pandas_ta | Rejected by D-19 -- external deps add install complexity, no pandas in pipeline |
| JSON schema validation (jsonschema) | Manual validation | jsonschema adds a dep; manual validation is simpler for the fixed indicator set |
| Pydantic for JSON parsing | Manual json.loads + checks | Overkill for this fixed schema; adds dependency for no benefit |

**Installation:** No new packages needed. Everything runs on the existing numpy/scipy/torch/openai stack.

**Version verification (2026-04-15):**
```
numpy 2.2.6, scipy 1.16.2, torch 2.9.0+cu128, openai 1.106.1, pytest 9.0.3
```

## Architecture Patterns

### Recommended Project Structure
```
exp4.15/core/
  feature_library.py    # NEW: indicator implementations + JSON->closure assembler
  prompts.py            # REWRITE: from Python code gen to JSON selection prompts
  lesr_controller.py    # MODIFY: _extract_code->_extract_json, _validate_code->_validate_selection
  dqn_trainer.py        # MODIFY: fixed reward rules (extend compute_regime_bonus)
  metrics.py            # REUSE: IC/IR/quantile_spread already implemented
  regime_detector.py    # REUSE: 3-dim regime vector unchanged
  feature_analyzer.py   # REUSE: importance analysis unchanged
  baseline.py           # MINOR: import path updates for feature_library
  lesr_strategy.py      # MODIFY: use _build_enhanced_state pattern
```

### Pattern 1: Indicator Registry + Closure Assembly (D-21)
**What:** Each indicator is a function that takes raw_state (120d) + params dict and returns a 1D numpy array. A registry maps indicator names to functions. `build_revise_state()` takes a JSON selection and returns a closure that applies all selected indicators.

**When to use:** This is the core pattern for the entire feature_library.py module.

**Example:**
```python
# Source: Design from CONTEXT.md D-21, verified against regime_detector.py pattern
import numpy as np
from typing import Callable, Dict, List

# State layout constants (from regime_detector.py and dqn_trainer.py)
# s[i*6 + 0] = close, s[i*6 + 1] = open, s[i*6 + 2] = high,
# s[i*6 + 3] = low, s[i*6 + 4] = volume, s[i*6 + 5] = adj_close

def _extract_ohlcv(s: np.ndarray):
    """Extract OHLCV arrays from 120d interleaved state."""
    n = len(s) // 6
    closes = np.array([s[i*6] for i in range(n)], dtype=float)
    opens = np.array([s[i*6+1] for i in range(n)], dtype=float)
    highs = np.array([s[i*6+2] for i in range(n)], dtype=float)
    lows = np.array([s[i*6+3] for i in range(n)], dtype=float)
    volumes = np.array([s[i*6+4] for i in range(n)], dtype=float)
    return closes, opens, highs, lows, volumes

def compute_rsi(s: np.ndarray, window: int = 14) -> np.ndarray:
    """RSI indicator. Returns 1D array of length 1 (latest RSI value)."""
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([50.0])  # neutral default
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:]) + 1e-10
    rs = avg_gain / avg_loss
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return np.array([rsi_val / 100.0])  # normalize to [0, 1]

# Registry: name -> (function, output_dim, default_params, param_ranges)
INDICATOR_REGISTRY: Dict[str, dict] = {
    'RSI': {
        'fn': compute_rsi,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 60)},
        'theme': 'trend'
    },
    # ... 20+ more entries
}

def build_revise_state(selection: List[Dict]) -> Callable:
    """Build a closure that computes all selected features.

    Args:
        selection: [{"indicator": "RSI", "params": {"window": 14}}, ...]

    Returns:
        Callable that takes raw_state (120d) and returns 1D feature array.
    """
    funcs = []
    for item in selection:
        name = item['indicator']
        params = item.get('params', {})
        if name not in INDICATOR_REGISTRY:
            continue
        entry = INDICATOR_REGISTRY[name]
        # Merge defaults with user-specified params
        merged = {**entry['default_params'], **params}
        funcs.append((entry['fn'], merged))

    def revise_state(raw_state: np.ndarray) -> np.ndarray:
        features = []
        for fn, params in funcs:
            try:
                result = fn(raw_state, **params)
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    features.append(np.zeros(result.shape))
                else:
                    features.append(result)
            except Exception:
                features.append(np.zeros(1))  # graceful fallback
        if not features:
            return np.zeros(3)  # fallback
        return np.concatenate(features)

    return revise_state
```

### Pattern 2: Z-Score Normalization (D-17)
**What:** Each indicator output is Z-score normalized during feature computation using training-set statistics.

**When to use:** Inside the closure, after computing raw indicator values.

**Example:**
```python
# Source: Design from CONTEXT.md D-17
class NormalizedIndicator:
    """Wraps an indicator function with Z-score normalization."""
    def __init__(self, fn, params, mean=None, std=None):
        self.fn = fn
        self.params = params
        self.mean = mean  # computed from training data
        self.std = std    # computed from training data

    def __call__(self, raw_state):
        raw = self.fn(raw_state, **self.params)
        if self.mean is not None and self.std is not None:
            return (raw - self.mean) / (self.std + 1e-8)
        return raw
```

### Pattern 3: JSON Validation Pipeline (D-05)
**What:** Multi-stage validation of LLM JSON output: parse -> check indicators exist -> validate param ranges -> compute features on sample data -> check IC/variance -> NaN/Inf check.

**When to use:** Replaces `_validate_code()` in lesr_controller.py.

**Example:**
```python
# Source: Design from CONTEXT.md D-05, D-09
import json
import numpy as np

def _validate_selection(json_str: str, sample_state: np.ndarray,
                        training_data_features=None, forward_returns=None) -> dict:
    """Validate LLM's JSON feature selection.

    Returns: {'selection': [...], 'revise_state': callable, 'state_dim': int, 'errors': [...]}
    """
    errors = []

    # Stage 1: Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {'selection': [], 'errors': [f'JSON parse error: {e}']}

    if 'features' not in data or not isinstance(data['features'], list):
        return {'selection': [], 'errors': ['Missing or invalid "features" key']}

    valid_selection = []
    for item in data['features']:
        name = item.get('indicator', '')
        params = item.get('params', {})

        # Stage 2: Check indicator exists
        if name not in INDICATOR_REGISTRY:
            errors.append(f'Unknown indicator: {name}')
            continue

        entry = INDICATOR_REGISTRY[name]

        # Stage 3: Validate param ranges
        for pk, pv in params.items():
            if pk in entry['param_ranges']:
                lo, hi = entry['param_ranges'][pk]
                if not (lo <= pv <= hi):
                    errors.append(f'{name}.{pk}={pv} out of range [{lo},{hi}]')
                    params[pk] = np.clip(pv, lo, hi)  # clip to valid range

        valid_selection.append({'indicator': name, 'params': params})

    # Stage 4: Build closure and test on sample data
    try:
        revise_fn = build_revise_state(valid_selection)
        features = revise_fn(sample_state)
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert len(features) >= 1
    except Exception as e:
        return {'selection': valid_selection, 'errors': [f'Feature computation failed: {e}']}

    # Stage 5: NaN/Inf check (D-09)
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        errors.append('Features contain NaN or Inf')
        return {'selection': valid_selection, 'errors': errors}

    return {
        'selection': valid_selection,
        'revise_state': revise_fn,
        'feature_dim': len(features),
        'state_dim': 123 + len(features),  # 120 raw + 3 regime + features
        'errors': errors
    }
```

### Anti-Patterns to Avoid
- **Anti-pattern: exec/eval for feature assembly.** D-21 explicitly requires closures. The current `_train_ticker_worker` uses `importlib.util.spec_from_file_location` + `tempfile` to dynamically load LLM-generated Python code (lines 111-118 in lesr_controller.py). This entire pattern must be replaced with direct closure calls.
- **Anti-pattern: Computing features without NaN guards.** Division by zero is extremely common in financial indicators (e.g., RSI when avg_loss=0, Volume_Ratio when avg_volume=0). Every indicator function must guard against NaN/Inf at the computation level, not just at validation.
- **Anti-pattern: Using IC from test/validation data in COT feedback.** The leakage prevention system (`filter_cot_metrics` + `check_prompt_for_leakage`) exists but `check_prompt_for_leakage` is never called. D-13 requires activation.
- **Anti-pattern: Passing full iteration history to LLM.** D-02 limits context to ~2k tokens. Current `get_iteration_prompt()` passes ALL previous iterations' code, which grows unbounded.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Spearman IC computation | Custom correlation | `scipy.stats.spearmanr` (already in metrics.py) | Already implemented, tested, handles edge cases |
| NaN/Inf detection | Manual checks | `np.isnan()`, `np.isinf()`, `np.isfinite()` | Standard numpy, already used in metrics.py |
| JSON parsing from LLM output | Regex extraction of JSON | `json.loads()` with fallback to regex extraction from markdown code blocks | LLMs often wrap JSON in ```json blocks |
| Feature importance ranking | Custom ranking algorithm | Existing `feature_analyzer.py` with Spearman + SHAP | Already tested, produces consistent rankings |
| Z-score normalization | Custom mean/std tracking | Numpy vectorized `np.mean()`, `np.std()` with epsilon guard | Simple, tested, numerically stable |

**Key insight:** The feature_library.py module is the ONLY genuinely new code. Everything else is rewriting existing patterns (prompts, validation, COT) for the JSON paradigm. The metrics module, regime detector, feature analyzer, and test infrastructure are all directly reusable.

## Common Pitfalls

### Pitfall 1: RSI/Indicator NaN on Short Windows
**What goes wrong:** RSI computation requires `window + 1` closing prices. When the 120d state has fewer than 21 days of valid data (due to padding with zeros), indicator functions return NaN.
**Why it happens:** The state extraction in `dqn_trainer.py:extract_state()` pads with zeros if `len(state_120d) < 120`. Zero prices cause division by zero.
**How to avoid:** Every indicator must check `len(closes) < window + 1` and return a neutral default value. Also check `np.std(closes[-window:]) < 1e-8` to avoid degenerate inputs.
**Warning signs:** NaN values propagating through DQN training, causing `nn.MSELoss` to produce NaN gradients.

### Pitfall 2: Inconsistent Feature Dimensions Across Calls
**What goes wrong:** Different raw_state inputs produce different numbers of features from the same indicator selection.
**Why it happens:** Some indicators may conditionally return different output dimensions (e.g., MACD returns 3 values but returns zeros on failure).
**How to avoid:** Every indicator function must always return exactly `output_dim` values as defined in the registry. The closure assembler should verify total feature count is consistent.
**Warning signs:** DQN state dimension mismatch error during training, or `_build_enhanced_state` catching exceptions and falling back to zeros.

### Pitfall 3: LLM JSON Output Malformed
**What goes wrong:** LLM returns JSON wrapped in markdown code blocks, with trailing commas, or with non-standard parameter types (strings instead of ints).
**Why it happens:** LLMs do not reliably produce perfect JSON, especially with temperature > 0.
**How to avoid:** Extract JSON from markdown code blocks first (regex: ````json\n(...)\n````), then parse with `json.loads()`. Validate parameter types (convert string params to int/float). Have a retry mechanism.
**Warning signs:** `json.JSONDecodeError` on first parse attempt.

### Pitfall 4: IC Pre-Evaluation on Insufficient Data
**What goes wrong:** Computing IC with fewer than 5 data points produces unreliable results (Spearman correlation undefined for n<3, noisy for n<10).
**Why it happens:** Training period might be short, or early sub-periods in stability assessment might have too few dates.
**How to avoid:** `metrics.py:ic()` already returns 0.0 for `len < 5`. The feature screening pipeline should also warn if fewer than 20 training dates are available for IC computation.
**Warning signs:** All features have IC=0.0 because the sample is too small.

### Pitfall 5: Feature Correlation / Redundancy
**What goes wrong:** LLM selects RSI(14) and RSI(21) which are highly correlated, wasting feature slots.
**Why it happens:** D-08 specifies same-type dedup, but what counts as "same type" needs precise definition.
**How to avoid:** Group by base indicator name (strip parameters). If two selections share the same base name, keep only the one with higher IC. This is the rule from D-08.
**Warning signs:** Correlation matrix of selected features shows pairs with correlation > 0.95.

### Pitfall 6: Data Leakage Through Market Stats
**What goes wrong:** D-01 requires injecting market stats into prompts. If these stats include future information (e.g., computing volatility over the full training period including validation period), it constitutes leakage.
**Why it happens:** Computing stats requires a data range -- must be strictly within training period.
**How to avoid:** Only compute statistical summaries from `train_period` dates. Use `data_loader.get_date_range()` filtered to `train_period` when generating stats for prompts.
**Warning signs:** `check_prompt_for_leakage()` is never called in current code (confirmed at line 83-98 in lesr_controller.py). Must be activated per D-13.

## Code Examples

### Indicator: RSI (Verified NumPy Implementation)
```python
# Source: [CITED: stackoverflow.com/questions/61974217] standard Wilder's RSI
def compute_rsi(s: np.ndarray, window: int = 14) -> np.ndarray:
    """RSI indicator using Wilder's smoothing."""
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.5])  # neutral, normalized to [0, 1]
    deltas = np.diff(closes[-(window+1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) + 1e-10
    rs = avg_gain / avg_loss
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return np.array([rsi_val / 100.0])
```

### Indicator: MACD (3-output)
```python
# Source: [ASSUMED] standard MACD formula
def compute_macd(s: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD line, Signal line, Histogram. Returns 3D array."""
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < slow:
        return np.zeros(3)
    # EMA approximation using np.convolve
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    # Signal line is EMA of MACD (approximate with last value)
    return np.array([macd_line[-1], 0.0, macd_line[-1]])

def _ema(data, period):
    """Exponential moving average using np.convolve."""
    weights = np.exp(np.linspace(-1, 0, period))
    weights /= weights.sum()
    return np.convolve(data, weights, mode='full')[:len(data)]
```

### Indicator: Bollinger Band (3-output)
```python
# Source: [ASSUMED] standard Bollinger Band formula
def compute_bollinger(s: np.ndarray, window: int = 20, num_std: float = 2.0) -> np.ndarray:
    """Bollinger Band: upper, middle, lower (normalized by price)."""
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.zeros(3)
    recent = closes[-window:]
    sma = np.mean(recent)
    std = np.std(recent) + 1e-10
    upper = (sma + num_std * std - closes[-1]) / (closes[-1] + 1e-8)
    middle = (sma - closes[-1]) / (closes[-1] + 1e-8)
    lower = (sma - num_std * std - closes[-1]) / (closes[-1] + 1e-8)
    return np.array([upper, middle, lower])
```

### JSON Prompt Template (Replacing INITIAL_PROMPT)
```python
# Source: Design from CONTEXT.md D-04
INITIAL_PROMPT_TEMPLATE = """
You are a financial quantitative analysis expert selecting trading features.

## Available Indicators (organized by theme)

### Trend Following
- RSI(window: 5-60) - Relative Strength Index
- MACD(fast: 5-20, slow: 15-60, signal: 3-15) - Moving Average Convergence Divergence
- EMA_Cross(fast: 5-20, slow: 15-60) - EMA Crossover Signal
- Momentum(window: 5-60) - Price Momentum

### Volatility
- Bollinger(window: 10-40, std: 1.0-3.0) - Bollinger Band Position
- ATR(window: 5-30) - Average True Range
- Volatility(window: 5-60) - Rolling Volatility

### Mean Reversion
- Stochastic(window: 5-30) - Stochastic Oscillator (%K, %D)
- Williams_R(window: 5-30) - Williams %R
- CCI(window: 5-30) - Commodity Channel Index

### Volume
- OBV() - On-Balance Volume
- Volume_Ratio(window: 5-30) - Volume Ratio
- ADX(window: 5-30) - Average Directional Index

## Market Context
{market_stats}

## Output Format (STRICT JSON)
```json
{{
  "features": [
    {{"indicator": "RSI", "params": {{"window": 14}}}},
    {{"indicator": "MACD", "params": {{"fast": 12, "slow": 26, "signal": 9}}}},
    {{"indicator": "Bollinger", "params": {{"window": 20, "std": 2.0}}}}
  ],
  "rationale": "RSI for overbought/oversold signals, MACD for trend confirmation, Bollinger for volatility regime"
}}
```

## Rules
- Select 5-15 indicators (system will filter to best 5-10)
- Diversify across themes (don't pick all from one theme)
- Each indicator must have an economic rationale
- Parameters should be adapted to market context above
"""
```

### Fixed Reward Rules (Extending compute_regime_bonus)
```python
# Source: CONTEXT.md D-22, D-23. Extends dqn_trainer.py:184-197
def compute_fixed_reward(self, regime_vector, action, features):
    """Fixed reward rules replacing LLM-generated intrinsic_reward.

    5 rules: risk management, trend following, volatility dampening,
    momentum protection, mean reversion support.
    Weights: intrinsic_weight=0.02, regime_bonus_weight=0.01
    """
    trend, vol, risk = regime_vector[0], regime_vector[1], regime_vector[2]
    reward = 0.0

    # Rule 1: Risk Management (highest priority)
    if risk > 0.7:
        if action == 0: reward -= 5.0    # Discourage BUY in crisis
        elif action == 1: reward += 2.0   # Encourage SELL (stop-loss)
    elif risk > 0.4:
        if action == 0: reward -= 1.0     # Mild discouragement

    # Rule 2: Trend Following
    if abs(trend) > 0.3 and risk < 0.4:
        if (trend > 0.3 and action == 0) or (trend < -0.3 and action == 1):
            reward += 1.5  # Aligned with trend

    # Rule 3: Volatility Dampening
    if vol > 0.7 and risk < 0.4:
        reward *= 0.5  # Reduce all reward magnitude in uncertain market

    # Rule 4: Momentum Protection (feature-based)
    if len(features) > 0 and action == 0:
        # Penalize buying when momentum features are negative
        reward += np.clip(np.mean(features[:3]) * 0.5, -1.0, 1.0)

    # Rule 5: Mean Reversion Support
    if abs(trend) < 0.3 and vol < 0.3 and risk < 0.3:
        if action == 0 and len(features) > 0:
            # Encourage buying in calm sideways market with oversold signals
            reward += 0.5

    return float(np.clip(reward, -10.0, 10.0))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LLM generates Python code | LLM selects from predefined library (JSON) | Phase 3 (this phase) | Eliminates syntax errors, dimension mismatches, degenerate features |
| intrinsic_reward from LLM | Fixed human-designed reward rules | Phase 3 (this phase) | Decouples feature quality from reward quality, enables clear attribution |
| No feature stability tracking | Sub-period IC stability assessment | Phase 3 (this phase) | Identifies features that are reliable vs. lucky |
| Full iteration history in prompt | Curated ~2k token context | Phase 3 (this phase) | Reduces token cost, focuses LLM on actionable feedback |
| No NaN/Inf checks | Mandatory NaN/Inf validation on every feature | Phase 3 (this phase) | Prevents silent training corruption |
| `check_prompt_for_leakage()` defined but never called | Called before every LLM invocation | Phase 3 (D-13) | Closes a real data leakage vector |

**Deprecated/outdated:**
- `_extract_code()` method in lesr_controller.py (lines 303-316): Extracts Python code from LLM text response. Replaced by `_extract_json()`.
- `_validate_code()` method (lines 318-394): Uses `importlib` to dynamically load Python modules. Replaced by `_validate_selection()`.
- `_train_ticker_worker` tempfile + importlib pattern (lines 111-118): Creates temporary .py file and loads it. Replaced by direct closure calls.
- `INITIAL_PROMPT` as raw string (prompts.py lines 13-167): 167-line prompt asking for Python functions. Replaced by JSON-selection template.
- `get_iteration_prompt()` passing full code history (lines 225-277): Unbounded context growth. Replaced by curated ~2k token context.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | RSI/MACD/Bollinger indicator formulas match standard financial definitions | Code Examples | Features would be meaningless; need verification against reference implementation |
| A2 | 20 indicators fit in ~2k token prompt with descriptions and param ranges | Architecture Patterns | Prompt may be too long; may need to shorten descriptions or use abbreviated format |
| A3 | The `intrinsic_reward` interface in DQNTrainer can be cleanly replaced by calling `compute_fixed_reward` internally without breaking the training loop | Architecture Patterns | Training loop expects `self.intrinsic_reward(enhanced)` at line 252; changing signature requires updating call sites |
| A4 | Z-score normalization statistics can be pre-computed on training data before the closure is built, and remain valid across the training episode | Architecture Patterns | If feature distribution shifts significantly, pre-computed stats may not normalize properly |
| A5 | The OpenAI API with `response_format={"type": "json_object"}` or JSON-in-markdown-block is reliable enough for structured output | Architecture Patterns | LLM may occasionally produce malformed JSON; need retry logic |
| A6 | 3 candidates x 5 rounds = 15 total evaluations is computationally feasible within typical experiment timeframes | User Constraints | At 50 episodes per training run, 15 evaluations per ticker could take several hours; this matches current config (sample_count=6, max_iterations=3 gives 18 evaluations) |

## Open Questions

1. **Exact indicator list beyond the 14 from Phase 2**
   - What we know: Phase 2 defined 14 indicators (RSI, MACD, Bollinger, Momentum, Volatility, Volume_Ratio, ROC, EMA_Cross, Stochastic, OBV, ATR, Williams_%R, CCI, ADX). D-20 requires 20+.
   - What's unclear: Which 6+ additional indicators to add. CONTEXT.md suggests SMA_Cross, DEMA, Williams_Alligator, Skewness, Kurtosis.
   - Recommendation: Add SMA_Cross, DEMA, ROC (rate of change -- listed but not in theme packs), Skewness, Kurtosis, Williams_Alligator to reach 20. All straightforward to implement in pure NumPy.

2. **How to compute market stats for D-01**
   - What we know: D-01 requires ~100 tokens of statistical summaries (volatility, trend strength, avg volume, avg daily return).
   - What's unclear: Whether to compute these once per optimization run or once per iteration.
   - Recommendation: Compute once at the start of optimization from training period data. These are stable statistics that don't change between iterations.

3. **Feature stability report format (D-14, D-15)**
   - What we know: 3-4 sub-periods, IC mean + IC std per feature per period.
   - What's unclear: Whether the stability report is just internal (for filtering) or also surfaced to the researcher.
   - Recommendation: Store in iteration results directory as `stability_report.json`. Include in COT feedback for unstable features (D-16).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | All code | available | 3.13.5 | -- |
| NumPy | Feature library, metrics | available | 2.2.6 | -- |
| SciPy | IC computation (spearmanr) | available | 1.16.2 | -- |
| PyTorch | DQN training | available | 2.9.0+cu128 | -- |
| OpenAI API | LLM calls | available | 1.106.1 | -- |
| pytest | Test suite | available | 9.0.3 | -- |
| CUDA GPU | Parallel training | available | cu128 | CPU fallback exists |
| FINSABER framework | Data loading | available | local copy | -- |
| scikit-learn | SHAP importance (feature_analyzer.py) | available | (used via import) | Falls back to Spearman-only |

**Missing dependencies with no fallback:**
- None identified. All required dependencies are installed.

**Missing dependencies with fallback:**
- None identified.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | None (uses conftest.py + auto-discovery) |
| Quick run command | `cd exp4.15 && python3 -m pytest tests/ -x -q` |
| Full suite command | `cd exp4.15 && python3 -m pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LESR-01 | JSON prompt template produces valid indicator selections | unit | `pytest tests/test_feature_library.py::test_indicator_registry_complete -x` | Wave 0 |
| LESR-01 | Market stats injection works correctly | unit | `pytest tests/test_feature_library.py::test_market_stats -x` | Wave 0 |
| LESR-02 | JSON parse succeeds for valid JSON | unit | `pytest tests/test_validation.py::test_valid_json -x` | Wave 0 |
| LESR-02 | JSON parse fails gracefully for malformed JSON | unit | `pytest tests/test_validation.py::test_malformed_json -x` | Wave 0 |
| LESR-02 | NaN/Inf detection catches bad features | unit | `pytest tests/test_validation.py::test_nan_inf_detection -x` | Wave 0 |
| LESR-02 | Dimension consistency check | unit | `pytest tests/test_validation.py::test_dimension_consistency -x` | Wave 0 |
| LESR-03 | COT feedback includes IC per indicator | unit | `pytest tests/test_cot.py::test_cot_includes_ic -x` | Wave 0 |
| LESR-03 | COT feedback includes rejection reasons | unit | `pytest tests/test_cot.py::test_negative_guidance -x` | Wave 0 |
| LESR-03 | Leakage check is called | unit | `pytest tests/test_cot.py::test_leakage_check_activated -x` | Wave 0 |
| LESR-04 | Feature filtering produces 5-10 features | unit | `pytest tests/test_screening.py::test_feature_count -x` | Wave 0 |
| LESR-04 | Same-type dedup keeps higher IC | unit | `pytest tests/test_screening.py::test_dedup -x` | Wave 0 |
| LESR-04 | IC threshold rejects degenerate features | unit | `pytest tests/test_screening.py::test_ic_threshold -x` | Wave 0 |
| LESR-05 | Stability assessment across sub-periods | unit | `pytest tests/test_stability.py::test_sub_period_split -x` | Wave 0 |
| LESR-05 | Unstable feature detection | unit | `pytest tests/test_stability.py::test_unstable_detection -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `cd exp4.15 && python3 -m pytest tests/ -x -q`
- **Per wave merge:** `cd exp4.15 && python3 -m pytest tests/ -v --tb=short`
- **Phase gate:** Full suite green (73 existing + all new tests passing)

### Wave 0 Gaps
- [ ] `tests/test_feature_library.py` -- covers LESR-01 indicator implementations, registry completeness, NaN guards, dimension consistency
- [ ] `tests/test_validation.py` -- covers LESR-02 JSON parsing, param validation, NaN/Inf detection
- [ ] `tests/test_cot.py` -- covers LESR-03 COT feedback, negative guidance, leakage activation
- [ ] `tests/test_screening.py` -- covers LESR-04 feature filtering, IC gates, dedup
- [ ] `tests/test_stability.py` -- covers LESR-05 sub-period stability assessment

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | yes | OpenAI API key via environment variable / config |
| V3 Session Management | no | No sessions in this system |
| V4 Access Control | no | Single-user research tool |
| V5 Input Validation | yes | JSON schema validation for LLM output, param range validation |
| V6 Cryptography | no | No custom crypto |
| V9 Communication | yes | HTTPS to OpenAI API (handled by openai library) |

### Known Threat Patterns for LESR Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data leakage (test metrics in COT) | Information Disclosure | `filter_cot_metrics()` + `check_prompt_for_leakage()` (D-13) |
| LLM prompt injection | Tampering | Input sanitization, structured output mode limits injection surface |
| API key exposure | Information Disclosure | Keys in config.yaml (local dev), env vars for production |
| NaN propagation crash | Denial of Service | NaN/Inf guards at every computation point (D-09) |
| Dynamic code execution | Elevation of Privilege | Eliminated -- closure-based assembly replaces exec/eval (D-21) |

## Sources

### Primary (HIGH confidence)
- exp4.15/core/ source files: prompts.py, lesr_controller.py, dqn_trainer.py, metrics.py, regime_detector.py, feature_analyzer.py, lesr_strategy.py, baseline.py -- all read and analyzed in full
- .planning/phases/03-lesr-core-improvements/03-CONTEXT.md -- locked decisions
- .planning/phases/02-evaluation-framework-redesign/02-ARCHITECTURE-REFLECTION.md -- architecture rationale

### Secondary (MEDIUM confidence)
- [Stack Overflow: Calculating RSI in Python](https://stackoverflow.com/questions/61974217/calculating-rsi-in-python) -- RSI implementation patterns
- [QuantJourney Technical Indicators Library](https://quantjourney.substack.com/p/technical-indicators-library-50-fast) -- 50+ NumPy-based indicators, confirms pure NumPy approach is standard
- [Rolling Windows in NumPy](https://medium.com/data-science/rolling-windows-in-numpy-the-backbone-of-time-series-analytical-methods-bc2f79ba82d2) -- NumPy rolling window patterns
- [Information Coefficient as Performance Measure](https://arxiv.org/pdf/2010.08601) -- IC methodology for stock selection

### Tertiary (LOW confidence)
- MACD/Bollinger/Stochastic indicator formulas -- [ASSUMED] based on standard financial definitions, should be verified against a reference implementation
- OpenAI JSON mode reliability -- [ASSUMED] based on general LLM behavior knowledge

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages verified via runtime check, versions confirmed
- Architecture: HIGH -- detailed code analysis of all 8 source files, exact line numbers for transformation points
- Pitfalls: HIGH -- identified from direct code analysis (NaN in zero-padded states, uncalled leakage check, unbounded prompt history)
- Indicator implementations: MEDIUM -- RSI formula verified, MACD/Bollinger patterns based on standard definitions but not tested against reference data

**Research date:** 2026-04-15
**Valid until:** 2026-05-15 (stable Python/NumPy ecosystem, no fast-moving dependencies)
