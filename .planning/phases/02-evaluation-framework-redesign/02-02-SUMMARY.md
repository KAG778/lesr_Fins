---
phase: 02-evaluation-framework-redesign
plan: 02
subsystem: evaluation, data-integrity
tags: [data-leakage, regime-stratified, cot-feedback, metrics, dqn-trainer]

# Dependency graph
requires:
  - phase: 02-evaluation-framework-redesign/01
    provides: metrics.py (sharpe_ratio, max_drawdown, etc.), extended evaluate() with factor_metrics
provides:
  - filter_cot_metrics() module-level function for stripping non-training metrics from results
  - check_prompt_for_leakage() utility for scanning rendered prompts for leaked metric names
  - Regime-stratified evaluation: evaluate() returns per-regime sharpe/max_dd/count
  - 33 new tests covering leakage prevention and regime-stratified metrics
affects: [02-03, lesr_controller, dqn_trainer, prompts]

# Tech tracking
tech-stack:
  added: [re module for pattern-based leakage detection]
  patterns: [whitelist-based metric filtering, regime-stratified return bucketing, NaN-safe regime classification]

key-files:
  created:
    - exp4.9_c/tests/test_leakage.py
    - exp4.9_c/tests/test_regime_eval.py
  modified:
    - exp4.9_c/lesr_controller.py
    - exp4.9_c/dqn_trainer.py

key-decisions:
  - "Whitelist approach for filter_cot_metrics: only sharpe, max_dd, total_return pass through, everything else stripped"
  - "Pattern-based check_prompt_for_leakage using regex, matching both underscore and space variants (win_rate / win rate)"
  - "NaN regime vectors default to 'sideways' classification (T-02-04 mitigation)"
  - "Regime returns bucketed by trend threshold: bull (>0.3), bear (<-0.3), sideways (otherwise)"

patterns-established:
  - "Leakage guard pattern: filter results before COT rendering, then scan rendered text as second check"
  - "Regime-stratified metrics: bucket daily returns by detect_regime trend, compute per-regime sharpe/max_dd"

requirements-completed: [EVAL-03, EVAL-04]

# Metrics
duration: 9min
completed: 2026-04-14
---

# Phase 02 Plan 02: COT Leakage Guard + Regime-Stratified Evaluation Summary

**filter_cot_metrics strips non-training metrics from COT feedback path; evaluate() returns per-regime Sharpe/MaxDD/count using detect_regime trend thresholds**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-14T18:25:48Z
- **Completed:** 2026-04-14T18:34:51Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- COT feedback path now guarantees LLM never sees factor_metrics, regime_metrics, sortino, calmar, win_rate, or any test/val metrics
- evaluate() computes and returns regime_metrics with bull/bear/sideways breakdown of Sharpe and MaxDD
- 33 new tests (21 leakage + 12 regime evaluation) all passing
- check_prompt_for_leakage provides post-rendering verification of COT prompt cleanliness

## Task Commits

Each task was committed atomically (TDD cycle):

1. **Task 1 RED: Add COT leakage tests** - `c69940e` (test)
2. **Task 1 GREEN: Add COT leakage guard** - `804734e` (feat)
3. **Task 2 RED: Add regime-stratified eval tests** - `eb00493` (test)
4. **Task 2 GREEN: Add regime-stratified evaluation** - `5944a1c` (feat)

## Files Created/Modified
- `exp4.9_c/lesr_controller.py` - Added filter_cot_metrics(), check_prompt_for_leakage(), modified _generate_cot_feedback() to filter results
- `exp4.9_c/dqn_trainer.py` - Modified evaluate() to bucket returns by regime and compute per-regime metrics
- `exp4.9_c/tests/test_leakage.py` - 21 tests for leakage prevention (strip/keep/detection)
- `exp4.9_c/tests/test_regime_eval.py` - 12 tests for regime-stratified metrics

## Decisions Made
- Whitelist approach for filter_cot_metrics: only sharpe, max_dd, total_return pass through. Stricter than a blacklist approach, ensuring any new metric added to evaluate() is automatically blocked unless explicitly allowed.
- check_prompt_for_leakage uses case-insensitive regex matching with support for both underscore and space variants (e.g., win_rate and "win rate").
- NaN regime vectors default to 'sideways' (per T-02-04 threat mitigation in plan).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed _generate_cot_feedback worst_trades access after filtering**
- **Found during:** Task 1 (COT leakage guard implementation)
- **Issue:** filter_cot_metrics strips all keys except sharpe/max_dd/total_return, but worst_trades collection tried to access r['trainer'] from filtered results
- **Fix:** Split loop to use filtered_results for scores (sent to LLM) and original results for trainer access (worst_trades, internal use only)
- **Files modified:** exp4.9_c/lesr_controller.py
- **Verification:** All 21 leakage tests pass

**2. [Rule 3 - Blocking] Fixed test mock to provide date-specific price data**
- **Found during:** Task 2 (regime-stratified evaluation tests)
- **Issue:** Test mock returned static price data for all dates, causing detect_regime to always classify as sideways regardless of price trend
- **Fix:** Changed get_data_by_date mock to use side_effect returning date-specific prices
- **Files modified:** exp4.9_c/tests/test_regime_eval.py
- **Verification:** All 12 regime evaluation tests pass

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes necessary for correct behavior. No scope creep.

## Issues Encountered
- None beyond the auto-fixed issues above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- COT feedback path is leak-proof: filter_cot_metrics + check_prompt_for_leakage provide double-layer protection
- Regime-stratified evaluation is ready for cross-stock/cross-window aggregation (EVAL-05 in plan 02-03)
- All new code tested with 33 unit tests

---
*Phase: 02-evaluation-framework-redesign*
*Completed: 2026-04-14*

## Self-Check: PASSED

All files verified:
- FOUND: exp4.9_c/lesr_controller.py
- FOUND: exp4.9_c/dqn_trainer.py
- FOUND: exp4.9_c/tests/test_leakage.py
- FOUND: exp4.9_c/tests/test_regime_eval.py
- FOUND: .planning/phases/02-evaluation-framework-redesign/02-02-SUMMARY.md

All commits verified:
- FOUND: c69940e (test RED leakage)
- FOUND: 804734e (feat GREEN leakage)
- FOUND: eb00493 (test RED regime)
- FOUND: 5944a1c (feat GREEN regime)
