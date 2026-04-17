---
phase: 01-diagnosis-infrastructure
plan: 01
subsystem: testing
tags: [logging, json-lines, feature-quality, pytest, numpy, scipy]

# Dependency graph
requires:
  - phase: none
    provides: "Independent module - no prior phase dependencies"
provides:
  - "StructuredLogger for JSON-lines run manifest logging (DIAG-05)"
  - "compute_feature_quality for per-feature quality metrics (DIAG-03)"
  - "pytest test infrastructure for diagnosis framework"
  - "13 passing tests covering both modules"
affects: [diagnosis-infrastructure]

# Tech tracking
tech-stack:
  added: [pytest, json, hashlib, numpy.var, scipy.stats.spearmanr]
  patterns: [JSON-lines manifest logging, config/code hashing, feature quality diagnostics]

key-files:
  created:
    - "exp4.7/diagnosis/__init__.py"
    - "exp4.7/diagnosis/structured_logger.py"
    - "exp4.7/diagnosis/feature_quality.py"
    - "exp4.7/diagnosis/tests/__init__.py"
    - "exp4.7/diagnosis/tests/conftest.py"
    - "exp4.7/diagnosis/tests/test_structured_logger.py"
    - "exp4.7/diagnosis/tests/test_feature_quality.py"
  modified: []

key-decisions:
  - "JSON-lines format for manifest logging (append-friendly, query-capable)"
  - "Feature quality metrics include variance, Spearman correlation, and information ratio"

patterns-established:
  - "Diagnosis modules in exp4.7/diagnosis/ with separate test files in exp4.7/diagnosis/tests/"
  - "sys.path.insert imports for test files (exp4.7 directory name contains dot)"
  - "Synthetic fixtures matching LESR pipeline state/reward shapes"

requirements-completed:
  - DIAG-05
  - DIAG-03

# Metrics
duration: 8min
completed: 2026-04-14
---

# Phase 1 Plan 01: Diagnosis Infrastructure Summary

**Test infrastructure, structured logging (DIAG-05), and feature quality metrics (DIAG-03) for LESR diagnosis**

## Performance

- **Duration:** 8 min
- **Tasks:** 1
- **Files modified:** 7

## Accomplishments
- StructuredLogger with JSON-lines manifest logging (run_id, config_hash, llm_code_hash, metrics)
- compute_feature_quality with per-feature variance, Spearman correlation, information ratio
- All 13 tests passing (7 structured_logger + 6 feature_quality)
- Handles degenerate features (zero variance) and NaN rewards without producing NaN

## Task Commits

Each task was committed atomically:

1. **Task 1: Create diagnosis infrastructure with structured logger and feature quality** - `114799d` (feat)

## Files Created/Modified
- `exp4.7/diagnosis/__init__.py` - Package init
- `exp4.7/diagnosis/structured_logger.py` - StructuredLogger class for JSON-lines run manifests
- `exp4.7/diagnosis/feature_quality.py` - compute_feature_quality function for feature quality diagnostics
- `exp4.7/diagnosis/tests/__init__.py` - Test package init
- `exp4.7/diagnosis/tests/conftest.py` - Shared pytest fixtures (synthetic states/rewards)
- `exp4.7/diagnosis/tests/test_structured_logger.py` - 7 tests for StructuredLogger
- `exp4.7/diagnosis/tests/test_feature_quality.py` - 6 tests for compute_feature_quality

## Decisions Made
- Used JSON-lines format for manifest files (append-friendly, no need to read entire file)
- Used hashlib for config/code hashing to detect duplicate runs

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tests passed on first run.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Diagnosis framework logging and feature quality modules ready for integration
- StructuredLogger can be used by RunManager (Plan 01-03) to track runs
- compute_feature_quality can be called per-run to identify bad LLM samples

---
*Phase: 01-diagnosis-infrastructure*
*Completed: 2026-04-14*
