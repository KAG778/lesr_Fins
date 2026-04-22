"""
V2 Fine-tune Transfer: Load V1's best_model.pt, fine-tune on test first 20%, eval last 80%.

Usage:
    python run_v2_transfer.py --window W1

直接复用 V1 实验的 best_code + best_model.pt，跳过 LESR 迭代，只做 Step 5 微调。
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from lesr_controller import LESRController
from code_sandbox import validate as sandbox_validate
from portfolio_env import PortfolioEnv
from ppo_agent import PPOAgent
from regime_detector import detect_market_regime
from metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']


def run_v2_transfer(config_path, window_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Paths
    v1_dir = Path(f'/home/wangmeiyi/AuctionNet/lesr/组合优化_ppo_策略迁移_v1/results/window_{window_name}')
    v2_dir = Path(f'/home/wangmeiyi/AuctionNet/lesr/组合优化_ppo_策略迁移_v2/results/window_{window_name}')
    v2_dir.mkdir(parents=True, exist_ok=True)

    # Load config from V1
    train_period = tuple(config['experiment']['train_period'])
    test_period = tuple(config['experiment']['test_period'])
    data_path = config['data']['pickle_file']
    transaction_cost = config['portfolio'].get('transaction_cost', 0.001)
    ppo_config = config.get('ppo', {})

    print(f"{'='*60}")
    print(f"V2 Fine-tune Transfer: {window_name}")
    print(f"Train: {train_period}, Test: {test_period}")
    print(f"Source: {v1_dir}")
    print(f"{'='*60}")

    # Load best model from V1
    model_path = v1_dir / 'best_model.pt'
    assert model_path.exists(), f"V1 model not found: {model_path}"

    # Find best iteration's code
    summary_path = v1_dir / 'summary.json'
    with open(summary_path) as f:
        summary = json.load(f)
    best_iter = summary.get('best_iteration', 1)

    code_path = v1_dir / f'iteration_{best_iter}' / 'code.py'
    with open(code_path) as f:
        best_code = f.read()

    config_path_iter = v1_dir / f'iteration_{best_iter}' / 'config.json'
    with open(config_path_iter) as f:
        iter_config = json.load(f)

    # Validate code to get closures
    code_sample = {'revise_state_fn': None, 'intrinsic_reward_fn': None}
    try:
        r = sandbox_validate(best_code)
        if r['ok']:
            code_sample = {
                'code': best_code,
                'revise_state_fn': r['revise_state'],
                'intrinsic_reward_fn': r['intrinsic_reward'],
                'feature_dim': r['feature_dim'],
                'state_dim': r['state_dim'],
            }
            print(f"  Code validated: feature_dim={r['feature_dim']}, state_dim={r['state_dim']}")
    except Exception as e:
        print(f"  Code validation failed: {e}")

    # Split test_period: first 20% for fine-tuning, last 80% for evaluation
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    all_dates = sorted(raw_data.keys())
    test_dates = [d for d in all_dates if test_period[0] <= d <= test_period[1]]
    split_idx = int(len(test_dates) * 0.2)
    finetune_period = (test_dates[0], test_dates[split_idx])
    eval_period = (test_dates[split_idx + 1], test_dates[-1])
    print(f"  Fine-tune: {finetune_period[0]} ~ {finetune_period[1]} ({split_idx + 1} days)")
    print(f"  Eval:      {eval_period[0]} ~ {eval_period[1]} ({len(test_dates) - split_idx - 1} days)")

    # --- LESR V2: Load model + fine-tune on finetune_period ---
    print(f"\n  --- LESR V2: Load + Fine-tune ({min(15, ppo_config.get('max_episodes', 50) // 3)} episodes) ---")
    env_ft = PortfolioEnv(
        data_path, config,
        revise_state_fn=code_sample.get('revise_state_fn'),
        detect_regime_fn=detect_market_regime,
        intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
        train_period=finetune_period,
        transaction_cost=transaction_cost,
    )
    agent = PPOAgent(
        state_dim=env_ft.state_dim,
        hidden_dim=ppo_config.get('hidden_dim', 256),
        use_twin_critic=ppo_config.get('use_twin_critic', True),
    )
    agent.load(str(model_path))
    print(f"  Loaded model from {model_path}")

    finetune_episodes = min(15, ppo_config.get('max_episodes', 50) // 3)
    for episode in range(finetune_episodes):
        state = env_ft.reset()
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            weights, log_prob = agent.select_action(state)
            next_state, reward, done, info = env_ft.step(weights)
            states.append(state)
            actions.append(weights)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            state = next_state
        if len(states) > 1:
            agent.update(states, actions, log_probs, rewards, dones, state)
        if (episode + 1) % 5 == 0:
            print(f"    Fine-tune episode {episode+1}/{finetune_episodes}")

    # Evaluate on eval_period
    test_result = _evaluate(agent, code_sample, data_path, config, eval_period, transaction_cost, "Test")

    # --- Baseline V2: Train on train_period + continue on finetune_period, eval last 80% ---
    print(f"\n  --- Baseline V2: Train on train + finetune on test first 20% ---")
    max_episodes = ppo_config.get('max_episodes', 50)

    env_train = PortfolioEnv(
        data_path, config,
        detect_regime_fn=detect_market_regime,
        train_period=train_period,
        transaction_cost=transaction_cost,
    )
    base_agent = PPOAgent(
        state_dim=env_train.state_dim,
        hidden_dim=ppo_config.get('hidden_dim', 256),
        actor_lr=ppo_config.get('actor_lr', 3e-4),
        critic_lr=ppo_config.get('critic_lr', 3e-4),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_epsilon=ppo_config.get('clip_epsilon', 0.2),
        entropy_coef=ppo_config.get('entropy_coef', 0.01),
        epochs_per_update=ppo_config.get('epochs_per_update', 10),
        batch_size=ppo_config.get('batch_size', 64),
        use_twin_critic=ppo_config.get('use_twin_critic', True),
        value_clip_epsilon=ppo_config.get('value_clip_epsilon', 0.2),
        dropout_rate=ppo_config.get('dropout_rate', 0.1),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        critic_weight_decay=ppo_config.get('critic_weight_decay', 1e-5),
    )
    print(f"  Phase 1: Training on {train_period[0]} ~ {train_period[1]} ({max_episodes} episodes)")
    for episode in range(max_episodes):
        state = env_train.reset()
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            weights, log_prob = base_agent.select_action(state)
            next_state, reward, done, info = env_train.step(weights)
            states.append(state)
            actions.append(weights)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            state = next_state
        if len(states) > 1:
            base_agent.update(states, actions, log_probs, rewards, dones, state)

    env_ft2 = PortfolioEnv(
        data_path, config,
        detect_regime_fn=detect_market_regime,
        train_period=finetune_period,
        transaction_cost=transaction_cost,
    )
    print(f"  Phase 2: Continue on {finetune_period[0]} ~ {finetune_period[1]} ({finetune_episodes} episodes)")
    for episode in range(finetune_episodes):
        state = env_ft2.reset()
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            weights, log_prob = base_agent.select_action(state)
            next_state, reward, done, info = env_ft2.step(weights)
            states.append(state)
            actions.append(weights)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            state = next_state
        if len(states) > 1:
            base_agent.update(states, actions, log_probs, rewards, dones, state)

    baseline_result = _evaluate(base_agent, {'revise_state_fn': None, 'intrinsic_reward_fn': None},
                                data_path, config, eval_period, transaction_cost, "Baseline")

    # Print comparison
    _print_comparison(test_result, baseline_result, test_period)

    # Save
    summary_extra = {
        'transfer_strategy': 'V2_finetune',
        'source': str(v1_dir),
        'finetune_period': list(finetune_period),
        'eval_period': list(eval_period),
        'finetune_episodes': finetune_episodes,
        'test_result': {k: v for k, v in test_result.items() if 'returns' not in k},
        'baseline_result': {k: v for k, v in baseline_result.items() if 'returns' not in k},
    }
    with open(v2_dir / 'final_comparison.json', 'w') as f:
        json.dump(summary_extra, f, indent=2, default=str)
    print(f"\n  Results saved to {v2_dir / 'final_comparison.json'}")


def _evaluate(agent, code_sample, data_path, config, period, transaction_cost, label):
    env = PortfolioEnv(
        data_path, config,
        revise_state_fn=code_sample.get('revise_state_fn'),
        detect_regime_fn=detect_market_regime,
        intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
        train_period=period,
        transaction_cost=transaction_cost,
    )
    state = env.reset()
    done = False
    returns = []
    weights_history = []
    while not done:
        weights, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(weights)
        returns.append(info.get('portfolio_return', 0))
        weights_history.append(info.get('weights', np.ones(6) / 6).copy())
        state = next_state

    ep_sharpe = sharpe_ratio(returns)
    ep_sortino = sortino_ratio(returns)
    ep_mdd = max_drawdown(returns)
    ep_calmar = calmar_ratio(returns)
    ep_return = (env.portfolio_value - 1.0) * 100
    avg_weights = np.mean(weights_history, axis=0)

    print(f"    [{label}] Sharpe={ep_sharpe:.3f}, Return={ep_return:.2f}%, MDD={ep_mdd:.2f}%")
    return {
        f'{label.lower()}_sharpe': ep_sharpe,
        f'{label.lower()}_sortino': ep_sortino,
        f'{label.lower()}_max_drawdown': ep_mdd,
        f'{label.lower()}_calmar': ep_calmar,
        f'{label.lower()}_total_return': ep_return,
        f'{label.lower()}_avg_weights': {**{TICKERS[i]: float(avg_weights[i]) for i in range(5)}, 'CASH': float(avg_weights[5])},
    }


def _print_comparison(test_result, baseline_result, test_period):
    def _get(result, suffix):
        for prefix in ['test', 'baseline', 'val']:
            v = result.get(f'{prefix}_{suffix}')
            if v is not None:
                return v
        return float('nan')

    print(f"\n{'='*60}")
    print(f"V2 COMPARISON (Eval: last 80% of test period)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'LESR+PPO':>12} {'PPO(tr+ft)':>12}")
    print(f"{'-'*20} {'-'*12} {'-'*12}")

    for label, suffix, fmt in [('Total Return', 'total_return', '%.2f%%'),
                                ('Sharpe Ratio', 'sharpe', '%.3f'),
                                ('Sortino Ratio', 'sortino', '%.3f'),
                                ('Max Drawdown', 'max_drawdown', '%.2f%%'),
                                ('Calmar Ratio', 'calmar', '%.3f')]:
        lv = _get(test_result, suffix)
        bv = _get(baseline_result, suffix)
        ls = fmt % lv if not np.isnan(lv) else 'N/A'
        bs = fmt % bv if not np.isnan(bv) else 'N/A'
        m = " *" if not np.isnan(lv) and not np.isnan(bv) and lv > bv else ""
        print(f"{label:<20} {ls:>12} {bs:>12}{m}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V2 Fine-tune Transfer')
    parser.add_argument('--window', type=str, required=True, help='Window name (W1-W5)')
    args = parser.parse_args()

    config_path = Path(__file__).parent / f'configs/config_{args.window}.yaml'
    run_v2_transfer(str(config_path), args.window)
