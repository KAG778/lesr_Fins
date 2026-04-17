---
phase: 01
slug: diagnosis-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-14
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (to be installed) |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `python -m pytest tests/diagnosis/ -x -q` |
| **Full suite command** | `python -m pytest tests/diagnosis/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/diagnosis/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/diagnosis/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | DIAG-01 | — | N/A | unit | `python -m pytest tests/diagnosis/ -x -q` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | DIAG-01 | — | N/A | unit | `python -m pytest tests/diagnosis/ -x -q` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | DIAG-02 | — | N/A | unit | `python -m pytest tests/diagnosis/ -x -q` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | DIAG-03 | — | N/A | unit | `python -m pytest tests/diagnosis/ -x -q` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 2 | DIAG-04 | — | N/A | unit | `python -m pytest tests/diagnosis/ -x -q` | ❌ W0 | ⬜ pending |
| 01-03-02 | 03 | 2 | DIAG-05 | — | N/A | unit | `python -m pytest tests/diagnosis/ -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/diagnosis/` — test directory with stubs for DIAG-01 through DIAG-05
- [ ] `tests/diagnosis/conftest.py` — shared fixtures (sample data paths, mock configs)
- [ ] `pip install pytest` — pytest framework

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Multi-run execution with 10+ seeds | DIAG-01 | Requires full LESR pipeline + GPU + LLM API | Launch via CLI, verify results directory populated |
| Statistical report visual inspection | DIAG-02 | Subjective quality of report output | Run report, verify p-values and CI displayed |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
