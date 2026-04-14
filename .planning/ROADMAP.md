# Roadmap: LESR 金融交易诊断与改进

## Overview

This roadmap transforms the LESR financial trading project from an unstable prototype into a rigorously validated research framework. The journey proceeds in three phases: first, build diagnostic tools to measure and decompose the instability problem; second, redesign the evaluation framework so that future improvements can be measured reliably; third, improve the LESR core (prompts, quality gates, feedback, feature filtering) using the diagnostic and evaluation infrastructure to verify each change.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Diagnosis Infrastructure** - Build tools to run, collect, and statistically analyze LESR vs DQN experiments
- [ ] **Phase 2: Evaluation Framework Redesign** - Redesign training/validation/testing to prevent leakage and enable robust multi-dimensional assessment
- [ ] **Phase 3: LESR Core Improvements** - Fix prompts, quality gates, COT feedback, and feature selection to stabilize LESR performance

## Phase Details

### Phase 1: Diagnosis Infrastructure
**Goal**: Researchers can run statistically rigorous experiments that quantify LESR instability and identify its sources
**Depends on**: Nothing (first phase)
**Requirements**: DIAG-01, DIAG-02, DIAG-03, DIAG-04, DIAG-05
**Success Criteria** (what must be TRUE):
  1. Researcher can launch 10+ independent runs of both DQN baseline and LESR with a single command, and all results are collected automatically
  2. Researcher can view a statistical comparison report (t-test or bootstrap p-value) showing whether LESR significantly outperforms DQN on Sharpe ratio
  3. Researcher can inspect per-run LLM-generated feature quality metrics (variance, return correlation, information ratio) to identify bad samples
  4. Researcher can see a variance decomposition report attributing instability to LLM sampling, DQN training, or data noise
  5. Researcher can retrieve the complete configuration, LLM output code, training curves, and final metrics for any past run from structured logs
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md -- Test infrastructure, structured logger (DIAG-05), and feature quality (DIAG-03)
- [x] 01-02-PLAN.md -- Statistical comparison (DIAG-02) and variance decomposition (DIAG-04)
- [x] 01-03-PLAN.md -- Run manager (DIAG-01), CLI entry point, and post-hoc analysis tool

### Phase 2: Evaluation Framework Redesign
**Goal**: Strategy evaluation is methodologically sound with walk-forward validation, multi-metric assessment (including factor evaluation IC/IR/Quantile Spread), leakage prevention, and market-regime awareness
**Depends on**: Phase 1 (diagnostic tools provide the measurement foundation)
**Baseline code**: exp4.9_c (Regime Detection, framework-level state assembly, safe reward, strict validation, worst-trade COT feedback)
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**Success Criteria** (what must be TRUE):
  1. Researcher can run walk-forward rolling-window experiments that train and test on sequentially advancing time windows, producing out-of-sample results
  2. Researcher can view a multi-metric evaluation report covering Sharpe, Sortino, max drawdown, Calmar ratio, and win rate, plus factor evaluation metrics (IC, IR, Quantile Spread) per feature dimension
  3. LLM iterative optimization feedback uses only training-set analysis, with no validation or test set information leaking into the prompt context
  4. Researcher can inspect strategy performance broken down by market regime (bull/bear/sideways) to identify regime-dependent weaknesses
  5. Researcher can generate a cross-stock, cross-window, cross-run comparison report that aggregates results into publication-ready tables
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md -- Metrics module (EVAL-02), extended evaluate() with factor_metrics, walk-forward compatibility verification (EVAL-01)
- [x] 02-02-PLAN.md -- Leakage guard (EVAL-03) and regime-stratified evaluation (EVAL-04)
- [x] 02-03-PLAN.md -- Cross-experiment aggregation and reporting (EVAL-05)

### Phase 3: LESR Core Improvements
**Goal**: LLM-generated features are economically meaningful, syntactically valid, dimensionally correct, and consistently outperform raw features across runs
**Depends on**: Phase 1 (diagnostic infrastructure to measure improvement), Phase 2 (robust evaluation to verify improvement is real)
**Baseline code**: exp4.9_c (Regime Detection, framework-level state assembly, safe reward, strict validation, worst-trade COT feedback)
**Requirements**: LESR-01, LESR-02, LESR-03, LESR-04, LESR-05
**Success Criteria** (what must be TRUE):
  1. LLM prompts produce features with explicit economic rationale (e.g., momentum, mean-reversion, volatility regime), verifiable in the generated code comments
  2. Every LLM-generated code sample passes automated checks for syntax correctness, output dimension matching, and numerical stability before being used in training
  3. COT feedback to LLM contains only high-confidence analysis results and explicit "do not do X" negative guidance, verifiable by inspecting the rendered prompt
  4. The feature set passed to DQN is filtered to 5-10 non-degenerate features, with degenerate features (zero variance, constant value) automatically rejected
  5. Researcher can view feature stability scores showing how consistently each feature performs across different time sub-periods
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD
- [ ] 03-03: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Diagnosis Infrastructure | 3/3 | Complete | 2026-04-14 |
| 2. Evaluation Framework Redesign | 0/3 | Planned | - |
| 3. LESR Core Improvements | 0/3 | Not started | - |
