# Phase 2: Evaluation Framework Redesign - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-14
**Phase:** 02-evaluation-framework-redesign
**Areas discussed:** Architecture Reflection, Feature Library Design, Reward Decoupling

---

## Architecture Reflection

| Option | Description | Selected |
|--------|-------------|----------|
| Keep current architecture | LLM generates both features and reward | |
| Decouple: fixed human reward + LLM features only | Eliminates attribution failure, LLM focuses on feature engineering | ✓ |
| Remove intrinsic_reward entirely | Pure price signal, no LLM reward | |
| Statistical auto-reward | Use feature correlations as reward | |

**User's choice:** Fixed human-designed reward + LLM generates features only
**Notes:** Root cause identified as intrinsic_reward coupling — when LLM changes both features and reward simultaneously, cannot attribute performance changes to either.

## Feature Engineering Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Free-form code generation (current) | LLM writes Python from scratch | |
| Structured feature library + LLM selection | LLM outputs JSON selecting indicators and params | ✓ |
| Statistical screening + LLM semantic labeling | Stats first, LLM adds meaning | |

**User's choice:** Structured feature library with LLM selecting combinations via JSON
**Notes:** Eliminates code generation instability. LLM focuses on judgment (which indicators suit this market?) not implementation.

## Claude's Discretion

- Feature library v1 exact scope and default parameters
- Fixed reward rule details and thresholds
- Report output format
- Sliding window sizes

## Deferred Ideas

- Custom feature proposal by LLM outside the library — future phase
- Ensemble DQN — independent, not in scope
- Adaptive growing feature library — Phase 3
- CPCV — v2 requirement
