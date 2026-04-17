---
phase: 01-diagnosis-infrastructure
reviewed: 2026-04-14T12:00:00Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - exp4.7/diagnosis/structured_logger.py
  - exp4.7/diagnosis/feature_quality.py
  - exp4.7/diagnosis/stats_reporter.py
  - exp4.7/diagnosis/variance_decomposition.py
  - exp4.7/diagnosis/run_manager.py
  - exp4.7/diagnosis/run_diagnosis.py
  - exp4.7/diagnosis/analyze_existing.py
  - exp4.7/diagnosis/config_diagnosis.yaml
  - exp4.7/diagnosis/tests/conftest.py
  - exp4.7/diagnosis/tests/test_structured_logger.py
  - exp4.7/diagnosis/tests/test_feature_quality.py
  - exp4.7/diagnosis/tests/test_stats_reporter.py
  - exp4.7/diagnosis/tests/test_variance_decomposition.py
  - exp4.7/diagnosis/tests/test_run_manager.py
findings:
  critical: 2
  warning: 5
  info: 3
  total: 10
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-04-14T12:00:00Z
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Reviewed the LESR Diagnosis Infrastructure module comprising 8 source files and 7 test files. The module implements structured logging, feature quality analysis, statistical comparison, variance decomposition, subprocess-isolated multi-run orchestration, and post-hoc analysis of existing results.

Two critical issues were found: unsafe deserialization of pickle files and a file handle leak in subprocess spawning. Five warnings include an incorrect between-group variance formula, fragile relative imports, shallow config copy causing shared mutable state, an unused import, and a missing YAML config field. Three informational items cover test-only import workarounds, a hardcoded API key in test fixtures, and incomplete test coverage for the full `launch_runs` pipeline.

## Critical Issues

### CR-01: Unsafe pickle deserialization of untrusted files

**File:** `exp4.7/diagnosis/analyze_existing.py:60-61`
**Issue:** `pickle.load(f)` is called on files from experiment directories without any restriction. Pickle deserialization of untrusted data can execute arbitrary code. The `load_iteration_results` function iterates over globbed paths and loads every `.pkl` file found, which is an injection vector if a malicious `.pkl` file is placed in a results directory.
**Fix:**
```python
# Option A: Use a restricted unpickler if the data types are known.
# Option B: Switch to a safer format (e.g., numpy .npz, JSON, or safetensors).
# Option C: At minimum, add a hash/signature verification before loading.
# For now, document the trust boundary:
import warnings
warnings.warn(
    "Loading pickle files from experiment directories. "
    "Only run this on directories you trust.",
    UserWarning,
    stacklevel=2,
)
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)
```

### CR-02: File handle leak in subprocess.Popen stdout/stderr redirection

**File:** `exp4.7/diagnosis/run_manager.py:151-155`
**Issue:** `open(run_dir / 'stdout.log', 'w')` and `open(run_dir / 'stderr.log', 'w')` are passed directly to `Popen` without being stored in a variable. The file handles are never closed, leaking file descriptors. If `num_runs` is large, this can exhaust the OS file descriptor limit before the processes complete.
**Fix:**
```python
# Store file handles and close them after process finishes
stdout_fh = open(run_dir / 'stdout.log', 'w')
stderr_fh = open(run_dir / 'stderr.log', 'w')
proc = subprocess.Popen(
    cmd, env=env,
    stdout=stdout_fh,
    stderr=stderr_fh,
)
# Close file handles in the parent process; the child has its own copies.
stdout_fh.close()
stderr_fh.close()
active_processes.append((proc, runs_info[-1]))
```

## Warnings

### WR-01: Between-group variance formula is incorrect for ANOVA decomposition

**File:** `exp4.7/diagnosis/variance_decomposition.py:78`
**Issue:** `between_group_variance = between_ss / (n_total - 1)` is not the standard ANOVA between-group variance estimate. The correct formula uses `between_ss / (n_total - 1)` only when computing total variance as `total_ss / (n_total - 1)`. For proper decomposition, between-group mean square should be `between_ss / (n_groups - 1)` and within-group mean square should be `within_ss / (n_total - n_groups)`. The current formula means `between_group_variance + within_group_variance` equals `total_ss / (n_total - 1)` (i.e., total variance), which is mathematically consistent but the individual components are not standard variance estimates. The test at `test_decomposition.py:44` only checks the sum, not the individual components, so the non-standard decomposition is not caught.
**Fix:**
```python
# Standard ANOVA decomposition:
between_group_variance = between_ss / (n_groups - 1) if n_groups > 1 else 0.0
within_group_variance = within_ss / (n_total - n_groups) if (n_total - n_groups) > 0 else 0.0
# Note: with this change, between + within will NOT sum to total_variance.
# The between_fraction should then be computed differently.
```
Alternatively, document that this is a non-standard "proportional" decomposition where the components partition the total sum of squares.

### WR-02: Fragile relative import of sibling module in run_manager.py

**File:** `exp4.7/diagnosis/run_manager.py:95`
**Issue:** `from structured_logger import StructuredLogger` is a bare relative import that works only when the script is executed from the `exp4.7/diagnosis/` directory or when that directory is on `sys.path`. If `run_manager` is imported as a proper submodule (e.g., `from diagnosis.run_manager import RunManager`), this import will fail. The same pattern appears in `run_diagnosis.py:60` and `analyze_existing.py:26-29`.
**Fix:**
```python
# Use explicit relative import for package usage:
from .structured_logger import StructuredLogger
# Or use the sys.path approach consistently with a guard:
try:
    from .structured_logger import StructuredLogger
except ImportError:
    from structured_logger import StructuredLogger
```

### WR-03: Shallow copy of config dict allows cross-run contamination

**File:** `exp4.7/diagnosis/run_manager.py:64`
**Issue:** `config = dict(self.base_config)` creates a shallow copy. If `self.base_config` contains nested mutable values (e.g., lists for tickers, dicts for output), mutations in one per-run config's nested structures will affect subsequent runs and the base config. The YAML config loaded at line 56-57 typically contains nested dicts (`output`, `experiment`, `dqn`, `llm`).
**Fix:**
```python
import copy
config = copy.deepcopy(self.base_config)
```

### WR-04: Unused import in run_manager.py

**File:** `exp4.7/diagnosis/run_manager.py:13`
**Issue:** `import hashlib` is imported but never used anywhere in the module.
**Fix:** Remove the unused import.

### WR-05: YAML config file missing the `analysis.confidence` field used by StatsReporter

**File:** `exp4.7/diagnosis/config_diagnosis.yaml:15`
**Issue:** The config defines `analysis.confidence: 0.95` and `analysis.bootstrap_resamples: 10000`, but neither `run_diagnosis.py` nor `analyze_existing.py` reads these values. `StatsReporter.compare_sharpe` hardcodes `confidence=0.95` and `n_resamples=10000` rather than reading from the config. If the user changes the config values, it will have no effect.
**Fix:** Either wire up the config values to the StatsReporter calls, or remove the misleading config fields.

## Info

### IN-01: sys.path manipulation for test imports

**Files:** `exp4.7/diagnosis/tests/test_structured_logger.py:17`, `exp4.7/diagnosis/tests/test_feature_quality.py:16`, `exp4.7/diagnosis/tests/test_run_manager.py:22`, `exp4.7/diagnosis/tests/test_stats_reporter.py:10`, `exp4.7/diagnosis/tests/test_variance_decomposition.py:10`
**Issue:** All test files use `sys.path.insert(0, ...)` to work around the dot in `exp4.7` which prevents standard Python imports. This is a pragmatic workaround but makes the test files fragile if directory structure changes.
**Fix:** Consider adding a `conftest.py` or `pyproject.toml` configuration to handle this centrally, or use a `pytest.ini` with `pythonpath` settings.

### IN-02: Hardcoded API key value in test fixture

**File:** `exp4.7/diagnosis/tests/conftest.py:81`
**Issue:** `sample_config` fixture contains `'api_key': 'test-key'`. While this is a test fixture and not a real key, it establishes a pattern. No real keys are present in the diagnosis source files themselves.
**Fix:** No action needed for test fixtures, but verify this value is never accidentally referenced as a real key.

### IN-03: analyze_existing.py uses broad except clauses without re-raising

**File:** `exp4.7/diagnosis/analyze_existing.py:67-68`, `exp4.7/diagnosis/analyze_existing.py:83-84`
**Issue:** `except Exception as e` in `load_iteration_results` silently swallows errors from corrupt pickle files, logging only a warning. This could mask real issues (e.g., file permission errors, out-of-memory errors on very large pickles). This is acceptable for a post-hoc analysis tool that needs to be robust, but worth noting.
**Fix:** Consider catching more specific exceptions (e.g., `pickle.UnpicklingError`, `EOFError`, `OSError`) instead of the broad `Exception`.

---

_Reviewed: 2026-04-14T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
