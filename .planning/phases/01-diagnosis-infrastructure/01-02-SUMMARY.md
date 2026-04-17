---
phase: 01-diagnosis-infrastructure
plan: 02
subsystem: testing
tags: [scipy, statistics, bootstrap, anova, variance-decomposition, t-test]

# Dependency graph
requires:
  - phase: none
    provides: "Independent module - no prior phase dependencies"
provides:
  - "StatsReporter for statistical comparison between LESR and DQN baselines"
  - "VarianceDecomposer for decomposing Sharpe ratio variance into LLM/DQN/data factors"
  - "11 passing tests covering both modules"
affects: [diagnosis-infrastructure, evaluation-redesign]

# Tech tracking
tech-stack:
  added: [scipy.stats.ttest_ind, scipy.stats.bootstrap, scipy.stats.mannwhitneyu, scipy.stats.f_oneway, scipy.stats.levene]
  patterns: [ANOVA-style variance decomposition, BCa bootstrap confidence interval, method-of-moments variance partitioning]

key-files:
  created:
    - "exp4.7/diagnosis/stats_reporter.py"
    - "exp4.7/diagnosis/variance_decomposition.py"
    - "exp4.7/diagnosis/tests/test_stats_reporter.py"
    - "exp4.7/diagnosis/tests/test_variance_decomposition.py"
  modified: []

key-decisions:
  - "Used sys.path imports in tests instead of dotted module paths (exp4.7 is not a valid Python package name due to dot)"
  - "Used ANOVA sum-of-squares decomposition for variance partitioning to ensure between + within = total exactly"

patterns-established:
  - "Diagnosis modules in exp4.7/diagnosis/ with separate test files in exp4.7/diagnosis/tests/"
  - "Statistical tests use scipy.stats directly with axis-aware statistic functions for bootstrap"
  - "Warning pattern: emit warning string when sample size insufficient (< 10 runs)"

requirements-completed:
  - DIAG-02
  - DIAG-04

# Metrics
duration: 12min
completed: 2026-04-14
---

# Phase 1 Plan 02: Diagnosis Infrastructure Summary

**Statistical comparison (t-test, bootstrap BCa, Mann-Whitney U) and ANOVA variance decomposition (LLM/DQN/data factors) for LESR diagnosis**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-14T14:30:00Z
- **Completed:** 2026-04-14T14:42:00Z
- **Tasks:** 1
- **Files modified:** 6

## Accomplishments
- StatsReporter with Welch's t-test, bootstrap BCa 95% CI, and Mann-Whitney U test for LESR vs DQN comparison
- VarianceDecomposer with three-factor decomposition (LLM sampling, DQN training, data/ticker) using ANOVA-style method of moments
- All 11 tests passing (5 stats_reporter + 6 variance_decomposition)
- Sample size warning when fewer than 10 runs provided

## Task Commits

Each task was committed atomically:

1. **Task 1: Create stats_reporter and variance_decomposition modules with tests** - `4e0f171` (feat)

## Files Created/Modified
- `exp4.7/diagnosis/__init__.py` - Package init
- `exp4.7/diagnosis/stats_reporter.py` - Statistical comparison between LESR and DQN (Welch's t-test, bootstrap BCa CI, Mann-Whitney U)
- `exp4.7/diagnosis/variance_decomposition.py` - ANOVA-based variance decomposition (LLM/DQN/data factors)
- `exp4.7/diagnosis/tests/__init__.py` - Test package init
- `exp4.7/diagnosis/tests/test_stats_reporter.py` - 5 tests for StatsReporter
- `exp4.7/diagnosis/tests/test_variance_decomposition.py` - 6 tests for VarianceDecomposer

## Decisions Made
- Used `sys.path` imports in tests instead of dotted module paths because `exp4.7` contains a dot which makes it an invalid Python package name
- Used ANOVA sum-of-squares (SS) decomposition for variance partitioning rather than naive `group_means.var()` subtraction, which can produce negative within-group variance

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Initial `from exp4.7.diagnosis.stats_reporter import StatsReporter` caused SyntaxError due to dot in directory name. Fixed by using `sys.path.insert` and direct module import in test files.
- Initial variance decomposition using `group_means.var()` subtraction produced negative within-group variance. Fixed by switching to proper ANOVA SS decomposition (between_ss + within_ss = total_ss).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Diagnosis framework core statistical modules ready for integration with data collection pipeline
- StatsReporter and VarianceDecomposer can be used independently or together for comprehensive diagnosis
- Ready for Plan 01-03 and subsequent data collection and analysis

---
*Phase: 01-diagnosis-infrastructure*
*Completed: 2026-04-14*
