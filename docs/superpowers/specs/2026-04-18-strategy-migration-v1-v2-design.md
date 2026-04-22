# Strategy Migration Design: V1 (Retrain) & V2 (Fine-tune)

**Date:** 2026-04-18
**Base version:** 组合优化_ppo
**Target:** 组合优化_ppo_策略迁移_v1, 组合优化_ppo_策略迁移_v2

## Problem

Current approach trains LESR on `train_period`, saves `best_model.pt`, then directly evaluates on `test_period`. The RL policy weights are overfitted to training market conditions and fail to generalize to different test-period markets.

LESR's unique advantage: LLM-generated code (`revise_state` + `intrinsic_reward`) represents transferable knowledge — feature extraction logic is universal, only RL weights are overfitted.

## Design Overview

| Version | Transferred | PPO Weights | Training Data |
|---------|------------|-------------|---------------|
| Current | best_code + best_model.pt | loaded directly | none (zero adaptation) |
| V1 | best_code + reward_config | train from scratch | test_period full |
| V2 | best_code + best_model.pt | fine-tune from loaded | test_period first 20% |

Baseline is adjusted to match: V1 baseline retrains on test_period, V2 baseline fine-tunes on test_period first 20%.

## V1: Code Migration + Retrain

### Step 5 Changes (lesr_controller.py lines 680-742)

1. Load `best_code` → re-validate via `code_sandbox.validate()` → get `revise_state_fn`, `intrinsic_reward_fn`
2. Load `best reward_config` → build `reward_rules_fn`
3. Create `PortfolioEnv` with `train_period=test_period` and LLM closures
4. Create new `PPOAgent` (random init, **do not load** best_model.pt)
5. Train PPO on test_period for `max_episodes` episodes (same count as training)
6. Evaluate trained agent on test_period (deterministic policy)
7. Baseline: train pure PPO (no LLM features) on test_period, evaluate on test_period
8. Print comparison table and save results

### Key Code Changes

Replace the current Step 5 block with a `_retrain_on_test()` method:

```python
def _retrain_on_test(self, code_sample, reward_config):
    """V1: Retrain PPO from scratch on test_period using LLM code."""
    env = PortfolioEnv(
        self.data_path, self.config,
        revise_state_fn=code_sample.get('revise_state_fn'),
        detect_regime_fn=detect_market_regime,
        intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
        reward_rules_fn=...,  # from reward_config
        train_period=self.test_period,
        transaction_cost=self.transaction_cost,
    )
    agent = PPOAgent(state_dim=env.state_dim, ...)
    for episode in range(self.max_episodes):
        # standard training loop
        ...
    return agent
```

### New Config Fields

```yaml
migration:
  version: 1
  finetune_episodes: null  # not used in V1
```

## V2: Fine-tune

### Step 5 Changes (lesr_controller.py lines 680-742)

1. Load `best_code` → re-validate → get closures (same as current)
2. Load `best_model.pt` into PPOAgent (same as current)
3. Split test_period: first 20% = fine-tune period, remaining = eval period
4. Create `PortfolioEnv` with `train_period=fine_tune_period` and LLM closures
5. Fine-tune loaded agent for `finetune_episodes` (default 10-15, fewer than full training)
6. Evaluate on full test_period (or remaining 80%)
7. Baseline: train pure PPO on train_period, load model, fine-tune on test_period first 20%
8. Print comparison and save results

### Key Code Changes

Add `_finetune_on_test()` method:

```python
def _finetune_on_test(self, code_sample, reward_config):
    """V2: Fine-tune loaded model on first 20% of test_period."""
    # Split test_period
    all_dates = sorted(self.raw_data.keys())
    test_dates = [d for d in all_dates
                  if self.test_period[0] <= d <= self.test_period[1]]
    split_idx = int(len(test_dates) * 0.2)
    finetune_period = (test_dates[0], test_dates[split_idx])

    # Load best model
    env = PortfolioEnv(..., train_period=finetune_period)
    agent = PPOAgent(state_dim=env.state_dim, ...)
    agent.load(str(model_path))

    # Fine-tune with fewer episodes
    for episode in range(self.finetune_episodes):  # 10-15
        ...
    return agent
```

### New Config Fields

```yaml
migration:
  version: 2
  finetune_episodes: 15
  finetune_ratio: 0.2  # first 20% of test_period for fine-tuning
  finetune_lr_scale: 0.1  # optional: reduced learning rate for fine-tuning
```

## Baseline Fairness

Both V1 and V2 must adjust baseline comparison:

- **V1 baseline**: Train pure PPO from scratch on test_period (same data as LESR retrain)
- **V2 baseline**: Train pure PPO on train_period, load model, fine-tune on test_period first 20%

This ensures fair comparison — both LESR and baseline get the same adaptation opportunity.

## File Structure

Each version is a full copy of 组合优化_ppo with only `lesr_controller.py` modified:

```
组合优化_ppo_策略迁移_v1/
  main.py                    # identical
  configs/config.yaml        # add migration.version: 1
  core/lesr_controller.py    # MODIFIED: Step 5 replaced with retrain logic
  core/*.py                  # identical
  data/                      # symlink or copy
  scripts/, tests/           # identical

组合优化_ppo_策略迁移_v2/
  main.py                    # identical
  configs/config.yaml        # add migration.version: 2, finetune params
  core/lesr_controller.py    # MODIFIED: Step 5 replaced with fine-tune logic
  core/*.py                  # identical
  data/                      # symlink or copy
  scripts/, tests/           # identical
```

## Implementation Scope

**Only one file changes per version**: `core/lesr_controller.py` (Step 5 block, ~60 lines).
All other files are identical copies from 组合优化_ppo.

## Data Leakage Consideration

V1 trains and evaluates on the same test_period. This is NOT data leakage — it is online learning / adaptation. The comparison is: "LESR features vs fixed features, under identical adaptation conditions." Both LESR and baseline get the same training data and evaluation protocol.
