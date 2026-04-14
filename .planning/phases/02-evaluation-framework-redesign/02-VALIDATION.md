---
phase: 02
slug: evaluation-framework-redesign
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-15
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml (if exists) or pytest.ini |
| **Quick run command** | `python -m pytest exp4.9_c/tests/ -x -q --tb=short` |
| **Full suite command** | `python -m pytest exp4.9_c/tests/ -v --tb=long` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest exp4.9_c/tests/ -x -q --tb=short`
- **After every plan wave:** Run `python -m pytest exp4.9_c/tests/ -v --tb=long`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | EVAL-01, EVAL-02 | T-02-01, T-02-02 | Validate returns length, return 0.0 for insufficient data | unit | `python -m pytest exp4.9_c/tests/test_metrics.py -x` | No -- W0 | pending |
| 02-01-02 | 01 | 1 | EVAL-01, EVAL-02 | T-02-02 | NaN propagation prevented via 0.0 defaults | unit | `python -m pytest exp4.9_c/tests/test_walk_forward.py -x` | No -- W0 | pending |
| 02-02-01 | 02 | 2 | EVAL-03 | T-02-03 | filter_cot_metrics strips non-training keys; check_prompt_for_leakage validates text | unit | `python -m pytest exp4.9_c/tests/test_leakage.py -x` | No -- W0 | pending |
| 02-02-02 | 02 | 2 | EVAL-04 | T-02-04 | classify_regime validates indexability; returns sideways on error | unit | `python -m pytest exp4.9_c/tests/test_regime_eval.py -x` | No -- W0 | pending |
| 02-03-01 | 03 | 2 | EVAL-05 | T-02-05, T-02-06 | pickle deserialization wrapped in try/except; file path from researcher filesystem | unit | `python -m pytest exp4.9_c/tests/test_cross_report.py -x` | No -- W0 | pending |
| 02-03-02 | 03 | 2 | EVAL-05 | — | N/A | unit | `python -m pytest exp4.9_c/tests/test_cross_report.py -x` (includes sliding_summary smoke test) | No -- W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `exp4.9_c/tests/__init__.py` -- test package init
- [ ] `exp4.9_c/tests/conftest.py` -- shared fixtures (sample returns, regime vectors, mock data_loader)
- [ ] `exp4.9_c/tests/test_metrics.py` -- stubs for EVAL-02 (Sortino, Calmar, WinRate with known inputs)
- [ ] `exp4.9_c/tests/test_walk_forward.py` -- stubs for EVAL-01 (extended evaluate() + walk-forward config verification)
- [ ] `exp4.9_c/tests/test_leakage.py` -- stubs for EVAL-03 (COT prompt inspection, guard mechanism)
- [ ] `exp4.9_c/tests/test_regime_eval.py` -- stubs for EVAL-04 (regime bucketing, per-regime metrics)
- [ ] `exp4.9_c/tests/test_cross_report.py` -- stubs for EVAL-05 (aggregation, report generation, sliding_summary smoke test)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Walk-forward produces sequential out-of-sample results | EVAL-01 | Requires real data and full training run | Run walk-forward on AMZN data, verify test windows don't overlap training |
| Cross-stock report renders publication-ready tables | EVAL-05 | Visual inspection of output format | Generate report and verify table alignment and metric completeness |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
