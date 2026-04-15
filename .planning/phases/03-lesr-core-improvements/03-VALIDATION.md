---
phase: 03
slug: lesr-core-improvements
status: draft
nyquist_compliant: false
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
| 03-01-01 | 01 | 1 | LESR-01 | — | N/A | unit | `pytest tests/test_feature_library.py -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | LESR-01, LESR-02 | — | N/A | unit | `pytest tests/test_feature_library.py -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | LESR-04 | — | N/A | unit | `pytest tests/test_feature_library.py -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | LESR-01 | — | N/A | unit | `pytest tests/test_prompt_json_mode.py -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | LESR-03 | — | N/A | unit | `pytest tests/test_cot_feedback.py -x` | ❌ W0 | ⬜ pending |
| 03-02-03 | 02 | 2 | LESR-03 | — | N/A | unit | `pytest tests/test_cot_feedback.py -x` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 2 | LESR-04 | — | N/A | unit | `pytest tests/test_screening.py -x` | ❌ W0 | ⬜ pending |
| 03-03-02 | 03 | 2 | LESR-05 | — | N/A | unit | `pytest tests/test_stability.py -x` | ❌ W0 | ⬜ pending |
| 03-03-03 | 03 | 2 | LESR-02 | — | N/A | unit | `pytest tests/test_fixed_reward.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `exp4.15/tests/test_feature_library.py` — stubs for LESR-01, LESR-02, LESR-04
- [ ] `exp4.15/tests/test_prompt_json_mode.py` — stubs for LESR-01
- [ ] `exp4.15/tests/test_cot_feedback.py` — stubs for LESR-03
- [ ] `exp4.15/tests/test_screening.py` — stubs for LESR-04
- [ ] `exp4.15/tests/test_stability.py` — stubs for LESR-05
- [ ] `exp4.15/tests/test_fixed_reward.py` — stubs for LESR-02
- [ ] `exp4.15/tests/conftest.py` — shared fixtures for feature data, mock LLM responses

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LLM prompt quality and economic rationale | LESR-01 | Requires LLM judgment | Inspect rendered prompt for market stats and theme pack guidance |
| Stability report readability | LESR-05 | Subjective quality | View generated stability_report.json, verify IC table is human-readable |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
