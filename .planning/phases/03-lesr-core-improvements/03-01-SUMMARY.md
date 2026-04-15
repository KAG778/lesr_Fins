---
phase: 03-lesr-core-improvements
plan: 01
subsystem: feature-engineering
tags: [numpy, financial-indicators, closure-pattern, z-score, registry, tdd]

# Dependency graph
requires: []
provides:
  - "INDICATOR_REGISTRY: 20 financial indicators with metadata (fn, output_dim, params, ranges, theme)"
  - "build_revise_state(): closure-based JSON-to-callable assembler (no exec/eval)"
  - "NormalizedIndicator: Z-score normalization wrapper for indicator outputs"
  - "_dedup_by_base_indicator(): same-type dedup keeping highest IC"
  - "_extract_ohlcv(): 120d interleaved state to separate OHLCV arrays"
affects: [03-02-PLAN, 03-03-PLAN, prompts.py, lesr_controller.py, dqn_trainer.py]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Closure-based feature assembly: build_revise_state returns inner function capturing indicator list"
    - "Registry pattern: indicator name -> metadata dict for lookup and validation"
    - "NaN/Inf guard: every indicator checks input length and returns finite defaults"
    - "Param clipping: build_revise_state clips user params to registered param_ranges"

key-files:
  created:
    - exp4.15/core/feature_library.py
    - exp4.15/tests/test_feature_library.py
  modified: []

key-decisions:
  - "20 indicators across 4 themes (trend:5, volatility:5, mean_reversion:3, volume:3, extended:4) -- meets D-20 requirement"
  - "EMA uses exponential decay weights via np.convolve for proper approximation"
  - "All indicators return finite numpy 1D arrays, never NaN/Inf"
  - "Closure captures funcs and output_dims lists for zero-allocation repeated calls"

patterns-established:
  - "Indicator function signature: compute_X(s: ndarray, **params) -> ndarray of shape (output_dim,)"
  - "Registry entry: {fn, output_dim, default_params, param_ranges, theme}"
  - "build_revise_state(selection) -> callable(raw_state) -> 1D ndarray"

requirements-completed: [LESR-01, LESR-02]

# Metrics
duration: 11min
completed: 2026-04-15
---

# Phase 03 Plan 01: Feature Library Summary

**20 NumPy financial indicators with INDICATOR_REGISTRY, closure-based JSON-to-callable assembler, Z-score normalization, and NaN/Inf safety guards**

## Performance

- **Duration:** 11 min
- **Started:** 2026-04-15T14:19:36Z
- **Completed:** 2026-04-15T14:30:22Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Implemented 20 pure-NumPy financial indicators across 4 themes (trend, volatility, mean_reversion, volume)
- Built INDICATOR_REGISTRY with complete metadata: function, output_dim, default_params, param_ranges, theme
- Created build_revise_state() closure assembler that converts JSON indicator selections to callable revise_state functions without exec/eval
- Implemented NormalizedIndicator wrapper for Z-score normalization
- Implemented _dedup_by_base_indicator() for same-type conflict resolution
- Full NaN/Inf safety on every indicator and in the closure fallback
- All 98 tests pass (73 existing + 25 new)

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD RED): Add failing tests** - `6a0ff62` (test)
2. **Task 1 (TDD GREEN): Implement feature_library** - `b9484b8` (feat)

_Note: Task 2 integration tests were included in the initial test file and pass with the GREEN implementation._

## Files Created/Modified
- `exp4.15/core/feature_library.py` - 763 lines: 20 indicator functions, INDICATOR_REGISTRY, build_revise_state(), NormalizedIndicator, _dedup_by_base_indicator()
- `exp4.15/tests/test_feature_library.py` - 379 lines: 25 tests covering registry, indicators, NaN safety, closure assembly, Z-score, integration

## Decisions Made
- EMA approximation uses exponential decay weights with np.convolve (standard approach, avoids manual loop)
- MACD signal line computed via EMA of MACD line (standard definition)
- Bollinger/ATR normalized by current price to get dimensionless values
- CCI normalized to [-1, 1] range by dividing by 200 (standard CCI ranges -200 to +200)
- ADX simplified to single-pass (vs multi-pass smoothing) for 20-day window compatibility
- Williams_Alligator uses simple SMAs at windows 13/8/5 (standard definition)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Worktree branch needed soft reset to correct base commit (f7fa81e)
- exp4.15 directory was untracked in main repo; copied to worktree for development

## Next Phase Readiness
- feature_library.py is ready for integration into lesr_controller.py (03-02-PLAN: JSON validation pipeline)
- build_revise_state() can replace the current exec/eval-based code generation
- INDICATOR_REGISTRY ready for prompt template construction (03-02-PLAN)
- All 4 theme packs populated and validated

## Self-Check: PASSED

- FOUND: .planning/phases/03-lesr-core-improvements/03-01-SUMMARY.md
- FOUND: exp4.15/core/feature_library.py
- FOUND: exp4.15/tests/test_feature_library.py
- FOUND: RED commit 6a0ff62
- FOUND: GREEN commit b9484b8

---
*Phase: 03-lesr-core-improvements*
*Completed: 2026-04-15*
