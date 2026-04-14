# LESR Architecture Reflection

**Date:** 2026-04-14
**Status:** Discussion concluded, direction agreed

## Problem Statement

The current LESR architecture has fundamental design issues that cannot be resolved by improving evaluation methodology (Phase 2) or prompt tuning (Phase 3) alone. The core issues are:

### 1. intrinsic_reward Coupling (Root Cause)

LLM simultaneously generates both `revise_state()` (features) and `intrinsic_reward()` (reward function). This creates:

- **Attribution failure**: Cannot determine whether poor performance is due to bad features or bad reward design
- **Wasted LLM capability**: LLM spends tokens on hard-coded reward rules (`if risk > 0.7: reward -= 40`) that humans can design better
- **Noisy feedback**: COT feedback can only give macro metrics (Sharpe), LLM cannot pinpoint whether to fix features or reward

### 2. Feature Quality Instability

LLM generates feature code from scratch with no data distribution information. Results:
- Degenerate features (zero variance)
- Thresholds misaligned with actual data distribution
- High variance across LLM samples

### 3. Ineffective Iteration Loop

Because both features and reward change simultaneously, iteration feedback is too vague for LLM to learn from. "Sharpe dropped from 1.2 to 0.8" gives no signal about what to change.

### 4. DQN Limitations (Amplified)

3 discrete actions, single DQN, sparse reward — these are independent issues but made worse by the coupling above.

## Causal Chain

```
Root: LLM role ambiguity (features + reward simultaneously)
  ├──► intrinsic_reward coupling → attribution failure
  ├──► Feature instability → LLM attention split + no data info
  ├──► Ineffective feedback → cannot isolate what to change
  └──► DQN limitations amplified → signal drowning in noise
```

## Decided Architecture Changes

### Change 1: Decouple intrinsic_reward (Fixed Human-Designed Reward)

**From:** LLM generates both `revise_state()` and `intrinsic_reward()`
**To:** intrinsic_reward is fixed as human-designed rules. LLM only generates features.

**Fixed reward design:**
- Regime-based rules (extend current `compute_regime_bonus`):
  - High risk + BUY → negative (stop-loss)
  - Strong trend + aligned features → positive (momentum)
  - High volatility → reduce reward magnitude (uncertainty)
- No LLM involvement in reward design

**Why not options A or C:**
- A (remove intrinsic_reward entirely): Too slow, price signal too sparse
- C (statistical auto-reward): Post-hoc correlation ≠ actionable signal, may conflict with price signal
- B (fixed human reward): Best tradeoff — preserves learning acceleration, eliminates coupling

### Change 2: Structured Feature Library + LLM Selection

**From:** LLM generates free-form Python code for features
**To:** LLM outputs structured JSON selecting from a predefined feature library

**Feature library (v1):**
```
RSI(window)              → 1D
MACD(fast, slow, signal) → 3D
Bollinger_Band(window, std) → 3D (upper, middle, lower)
Momentum(window)         → 1D
Volatility(window)       → 1D
Volume_Ratio(window)     → 1D
ROC(window)              → 1D (rate of change)
EMA_Cross(fast, slow)    → 1D
Stochastic_Osc(window)   → 2D (%K, %D)
OBV()                    → 1D (on-balance volume)
ATR(window)              → 1D (average true range)
Williams_%R(window)      → 1D
CCI(window)              → 1D (commodity channel index)
ADX(window)              → 1D (average directional index)
```

**LLM output format:**
```json
{
  "features": [
    {"indicator": "RSI", "params": {"window": 14}},
    {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
    ...
  ],
  "rationale": "RSI for overbought/oversold, MACD for trend..."
}
```

**Why this approach:**
1. Eliminates code generation instability (no syntax errors, no dimension mismatch, no degenerate features)
2. LLM focuses on judgment ("which indicators suit this market?") not implementation ("how to compute RSI")
3. Search space is controllable and finite
4. Feedback can be precise per-indicator ("RSI(14) IC=0.12, Momentum(10) IC=0.35")
5. Innovation space preserved via rationale + optional custom feature proposal

## Impact on ROADMAP

These changes affect Phase 2 (Evaluation) and Phase 3 (LESR Core):

- Phase 2 should evaluate under the **new architecture** (fixed reward + structured features), not the old one
- Phase 3's scope changes from "improve prompts and feedback" to "build feature library + structured LLM interface"
- The three-phase structure remains valid, but implementation details shift significantly

## Open Questions

1. Feature library v1 scope — 10-15 indicators enough? Which ones are must-haves?
2. Should LLM be allowed to propose new indicator types outside the library?
3. Fixed reward rule details — how many rules, what thresholds?
4. How to handle the transition from exp4.9_c to the new architecture?

---

*Discussion participants: User + Claude*
*Date: 2026-04-14*
