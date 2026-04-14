---
phase: 01-diagnosis-infrastructure
plan: 03
subsystem: infra
tags: [subprocess, seed-isolation, cli, post-hoc-analysis, manifest-jsonl, yaml-config]

# Dependency graph
requires:
  - phase: 01-diagnosis-infrastructure (01-01)
    provides: StructuredLogger for run manifest, feature_quality for diagnostics
  - phase: 01-diagnosis-infrastructure (01-02)
    provides: StatsReporter for statistical comparison, VarianceDecomposer for variance analysis
provides:
  - RunManager: subprocess-isolated multi-run orchestrator with seed control
  - run_diagnosis.py: CLI entry point for diagnosis experiments
  - analyze_existing.py: post-hoc analysis of past experiment results
  - config_diagnosis.yaml: default diagnosis experiment configuration
affects: [02-evaluation-redesign, 03-lesr-core-improvements]

# Tech tracking
tech-stack:
  added: []
  patterns: [subprocess-isolated-runs, seed-determinism, manifest-jsonl-tracking]

key-files:
  created:
    - exp4.7/diagnosis/run_manager.py
    - exp4.7/diagnosis/run_diagnosis.py
    - exp4.7/diagnosis/analyze_existing.py
    - exp4.7/diagnosis/config_diagnosis.yaml
    - exp4.7/diagnosis/tests/test_run_manager.py
  modified: []

key-decisions:
  - "Sibling imports via sys.path.insert for exp4.7 (dot in directory name breaks standard Python imports)"
  - "analyze_existing handles both old-format (result_W*) and new-format (run_XXX/) directories"

patterns-established:
  - "RunManager pattern: per-run config YAML + seed.txt + isolated directory"
  - "CLI entry point: argparse with --analyze-only flag for offline analysis"

requirements-completed: [DIAG-01]

# Metrics
duration: 8min
completed: 2026-04-14
---

# Phase 01 Plan 03: Run Orchestration Summary

**Subprocess-isolated multi-run orchestrator with seed control, CLI entry point, and post-hoc analysis tool for existing experiment directories**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-14T10:33:36Z
- **Completed:** 2026-04-14T10:41:30Z
- **Tasks:** 2
- **Files modified:** 5 (all new)

## Accomplishments
- RunManager launches N independent LESR/DQN runs with subprocess isolation, unique seeds, and max_parallel rate limiting
- CLI supports --analyze-only mode for post-hoc analysis without running new experiments
- analyze_existing.py successfully loads and reports on existing result_W1_test2019 directory (64 results across 4 tickers, 3 iterations)
- All 31 tests pass across the full diagnosis test suite (Plan 01 + Plan 02 + Plan 03)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create RunManager with subprocess isolation and seed control** - `80a83df` (feat)
2. **Task 2: Create CLI entry point and post-hoc analysis tool** - `b30f4da` (feat)

## Files Created/Modified
- `exp4.7/diagnosis/run_manager.py` - Subprocess-isolated multi-run orchestrator with seed control and manifest logging
- `exp4.7/diagnosis/tests/test_run_manager.py` - 7 tests for RunManager (config creation, seed uniqueness, directories, manifest, seed sequence)
- `exp4.7/diagnosis/run_diagnosis.py` - CLI entry point with --config, --num-runs, --output-dir, --mode, --analyze-only, --report-type
- `exp4.7/diagnosis/analyze_existing.py` - Post-hoc analysis of old-format and new-format result directories
- `exp4.7/diagnosis/config_diagnosis.yaml` - Default diagnosis experiment configuration

## Decisions Made
- Used sibling module imports via sys.path.insert because exp4.7 (dot in directory name) breaks standard Python package imports
- analyze_existing.py handles both old-format (result_W*_test*) and new-format (run_XXX/) directories for backward compatibility
- Mocked subprocess.Popen in tests to avoid launching actual GPU training

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- VarianceDecomposer produces NaN for DQN variance fraction when all runs share the same dqn_seed (0) -- this is a pre-existing issue from Plan 02, not caused by this plan's changes. Out of scope to fix here.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Complete diagnosis infrastructure: structured logging (Plan 01), statistical analysis (Plan 02), run orchestration (Plan 03)
- Researcher can launch multi-run experiments with: `python exp4.7/diagnosis/run_diagnosis.py --config exp4.7/config_W1.yaml --num-runs 10`
- Researcher can analyze existing results with: `python exp4.7/diagnosis/analyze_existing.py --results-dir exp4.7/result_W1_test2019 --report-type all`
- Phase 01 complete; ready for Phase 02 (Evaluation Redesign)

## Self-Check: PASSED

All 6 created files verified present. Both task commits (80a83df, b30f4da) verified in git log.

---
*Phase: 01-diagnosis-infrastructure*
*Completed: 2026-04-14*
