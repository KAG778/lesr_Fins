---
phase: 03
slug: lesr-core-improvements
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-15
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `exp4.15/tests/conftest.py` |
| **Quick run command** | `cd exp4.15 && python -m pytest tests/ -x -q --tb=short` |
| **Full suite command** | `cd exp4.15 && python -m pytest tests/ -v --tb=long` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd exp4.15 && python -m pytest tests/ -x -q --tb=short`
- **After every plan wave:** Run `cd exp4.15 && python -m pytest tests/ -v --tb=long`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | LESR-01, LESR-02 | T-03-01, T-03-03 | Params clipped to ranges; no exec/eval | unit | `cd exp4.15 && pytest tests/test_feature_library.py -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | LESR-01, LESR-02 | T-03-02 | NaN/Inf guards on all indicators | unit | `cd exp4.15 && pytest tests/test_feature_library.py -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 2 | LESR-01, LESR-03 | T-03-04 | Market stats from training data only; COT filtered | unit | `cd exp4.15 && pytest tests/test_prompts.py -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 2 | LESR-02, LESR-04, LESR-05 | T-03-05, T-03-06 | Multi-stage validation; IC/variance gates; stability checks | unit | `cd exp4.15 && pytest tests/test_validation.py tests/test_screening.py tests/test_stability.py -x` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 3 | LESR-02 | — | Fixed reward replaces LLM-generated reward | unit | `cd exp4.15 && pytest tests/ -x -q` | ❌ W0 | ⬜ pending |
| 03-03-02 | 03 | 3 | LESR-01, LESR-03 | T-03-07, T-03-09, T-03-11 | Leakage check active; JSON mode end-to-end | unit | `cd exp4.15 && pytest tests/test_controller_integration.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `exp4.15/tests/test_feature_library.py` — stubs for LESR-01, LESR-02 (Plan 01 Tasks 1+2)
- [ ] `exp4.15/tests/test_prompts.py` — stubs for LESR-01, LESR-03 (Plan 02 Task 1)
- [ ] `exp4.15/tests/test_validation.py` — stubs for LESR-02 (Plan 02 Task 2)
- [ ] `exp4.15/tests/test_screening.py` — stubs for LESR-04 (Plan 02 Task 2)
- [ ] `exp4.15/tests/test_stability.py` — stubs for LESR-05 (Plan 02 Task 2)
- [ ] `exp4.15/tests/test_controller_integration.py` — stubs for LESR-01, LESR-03 (Plan 03 Task 2)
- [ ] `exp4.15/tests/conftest.py` — shared fixtures for feature data, mock LLM responses

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LLM prompt quality and economic rationale | LESR-01 | Requires LLM judgment | Inspect rendered prompt for market stats and theme pack guidance |
| Stability report readability | LESR-05 | Subjective quality | View generated stability_report.json, verify IC table is human-readable |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
