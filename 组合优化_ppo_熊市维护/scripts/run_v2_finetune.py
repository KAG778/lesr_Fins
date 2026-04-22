"""
V2: Load V1 best model + best code, fine-tune on test period first 20%, evaluate last 80%.
"""

import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'core'))

import yaml
from feature_library import build_revise_state_from_code
from portfolio_features import build_portfolio_features
from regime_detector import detect_market_regime
from portfolio_env import PortfolioEnv
from ppo_agent import PPOAgent
from metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio
from code_sandbox import sandbox_validate

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']


def run_v2_finetune(config_path: str, v1_result_dir: str, experiment_name: str = 'v2_finetune'):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results_dir = PROJECT_DIR / 'results' / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = config.get('experiment', {})
    train_period = tuple(exp_cfg.get('train_period'))
    test_period = tuple(exp_cfg.get('test_period'))

    ppo_cfg = config.get('ppo', {})
    max_episodes = ppo_cfg.get('max_episodes', 50)
    hidden_dim = ppo_cfg.get('hidden_dim', 256)
    transaction_cost = config.get('portfolio', {}).get('transaction_cost', 0.001)
    data_path = config.get('data', {}).get('pickle_file', 'data/portfolio_5stocks.pkl')

    # ========== Load V1 best code + model ==========
    v1_dir = Path(v1_result_dir)
    summary = json.load(open(v1_dir / 'summary.json'))
    best_iter = summary['best_iteration']
    iter_dir = v1_dir / f'iteration_{best_iter}'

    # Load code
    with open(iter_dir / 'code.py') as f:
        best_code = f.read()

    code_result = sandbox_validate(best_code)
    if not code_result['ok']:
        print(f"ERROR: V1 best code invalid: {code_result.get('error')}")
        return None

    revise_fn = code_result['revise_state']
    intrinsic_reward_fn = code_result.get('intrinsic_reward')
    feature_dim = code_result['feature_dim']
    state_dim_from_code = code_result['state_dim']

    # Load reward config
    iter_config = json.load(open(iter_dir / 'config.json'))
    reward_rules = iter_config.get('reward_rules', [])

    print("=" * 60)
    print("V2 Fine-tune Transfer")
    print(f"V1 source: {v1_dir}")
    print(f"Best iteration: {best_iter}")
    print(f"V1 train Sharpe: {summary['best_train_sharpe']:.3f}")
    print(f"Feature dim: {feature_dim}, State dim: {state_dim_from_code}")
    print(f"Reward rules: {[r['rule'] for r in reward_rules]}")
    print(f"Train period: {train_period}")
    print(f"Test period:  {test_period}")
    print("=" * 60)

    # ========== Split test_period ==========
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    all_dates = sorted(raw_data.keys())
    test_dates = [d for d in all_dates if test_period[0] <= d <= test_period[1]]
    split_idx = int(len(test_dates) * 0.2)
    finetune_period = (test_dates[0], test_dates[split_idx])
    eval_period = (test_dates[split_idx + 1], test_dates[-1])

    print(f"Fine-tune (20%): {finetune_period[0]} ~ {finetune_period[1]} ({split_idx + 1} days)")
    print(f"Eval (80%):      {eval_period[0]} ~ {eval_period[1]} ({len(test_dates) - split_idx - 1} days)")

    # ========== Build reward function ==========
    from reward_rules import build_reward_rules
    reward_fn = build_reward_rules(reward_rules)

    # ========== Create env + load V1 model ==========
    port_fn = build_portfolio_features([
        {'indicator': 'momentum_rank', 'params': {'window': 20}},
        {'indicator': 'portfolio_volatility', 'params': {'window': 20}},
    ])

    finetune_env = PortfolioEnv(
        data_path, config,
        revise_state_fn=revise_fn,
        portfolio_features_fn=port_fn,
        detect_regime_fn=detect_market_regime,
        intrinsic_reward_fn=intrinsic_reward_fn,
        train_period=finetune_period,
        transaction_cost=transaction_cost,
    )

    state_dim = finetune_env.state_dim
    print(f"\nState dim: {state_dim}")

    agent = PPOAgent(
        state_dim=state_dim, hidden_dim=hidden_dim,
        actor_lr=ppo_cfg.get('actor_lr', 3e-4),
        critic_lr=ppo_cfg.get('critic_lr', 3e-4),
        gamma=ppo_cfg.get('gamma', 0.99),
        gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
        clip_epsilon=ppo_cfg.get('clip_epsilon', 0.2),
        entropy_coef=ppo_cfg.get('entropy_coef', 0.01),
        epochs_per_update=ppo_cfg.get('epochs_per_update', 10),
        batch_size=ppo_cfg.get('batch_size', 64),
        use_twin_critic=ppo_cfg.get('use_twin_critic', True),
        value_clip_epsilon=ppo_cfg.get('value_clip_epsilon', 0.2),
        dropout_rate=ppo_cfg.get('dropout_rate', 0.1),
        max_grad_norm=ppo_cfg.get('max_grad_norm', 0.5),
        critic_weight_decay=ppo_cfg.get('critic_weight_decay', 1e-5),
    )

    # Load V1 best model
    model_path = v1_dir / 'best_model.pt'
    agent.load(str(model_path))
    print(f"Loaded V1 model from {model_path}")

    # ========== Fine-tune on test first 20% ==========
    finetune_episodes = min(15, max_episodes // 3)
    print(f"\n--- Fine-tuning ({finetune_episodes} episodes) ---")

    all_rewards = []
    all_returns = []
    for episode in range(finetune_episodes):
        state = finetune_env.reset()
        episode_reward = 0
        episode_returns = []
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            weights, log_prob = agent.select_action(state)
            next_state, reward, done, info = finetune_env.step(weights)
            states.append(state)
            actions.append(weights)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            episode_reward += reward
            episode_returns.append(info.get('portfolio_return', 0))
            state = next_state
        if len(states) > 1:
            agent.update(states, actions, log_probs, rewards, dones, state)
        all_rewards.append(episode_reward)
        all_returns.extend(episode_returns)
        if (episode + 1) % 5 == 0:
            print(f"  Fine-tune episode {episode+1}/{finetune_episodes}: reward={episode_reward:.2f}")

    ft_sharpe = sharpe_ratio(all_returns) if all_returns else 0
    ft_return = (finetune_env.portfolio_value - 1.0) * 100
    print(f"Fine-tune: Sharpe={ft_sharpe:.3f}, Return={ft_return:.2f}%")

    # ========== Evaluate on test last 80% ==========
    eval_env = PortfolioEnv(
        data_path, config,
        revise_state_fn=revise_fn,
        portfolio_features_fn=port_fn,
        detect_regime_fn=detect_market_regime,
        intrinsic_reward_fn=intrinsic_reward_fn,
        train_period=eval_period,
        transaction_cost=transaction_cost,
    )
    state = eval_env.reset()
    done = False
    test_returns = []
    weights_history = []
    while not done:
        weights, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = eval_env.step(weights)
        test_returns.append(info.get('portfolio_return', 0))
        weights_history.append(info.get('weights', np.ones(6)/6).copy())
        state = next_state

    test_sharpe = sharpe_ratio(test_returns)
    test_sortino = sortino_ratio(test_returns)
    test_mdd = max_drawdown(test_returns)
    test_calmar = calmar_ratio(test_returns)
    test_return = (eval_env.portfolio_value - 1.0) * 100
    weights_arr = np.array(weights_history)
    avg_weights = weights_arr.mean(axis=0)

    print(f"\n{'='*60}")
    print(f"V2 Eval Results ({eval_period[0]} ~ {eval_period[1]})")
    print(f"{'='*60}")
    print(f"  Total Return:  {test_return:.2f}%")
    print(f"  Sharpe Ratio:  {test_sharpe:.3f}")
    print(f"  Sortino Ratio: {test_sortino:.3f}")
    print(f"  Max Drawdown:  {test_mdd:.2f}%")
    print(f"  Calmar Ratio:  {test_calmar:.3f}")

    result = {
        'method': 'V2_finetune',
        'v1_source': str(v1_dir),
        'best_iteration': best_iter,
        'finetune_period': finetune_period,
        'eval_period': eval_period,
        'finetune_sharpe': ft_sharpe,
        'finetune_return': ft_return,
        'test_sharpe': test_sharpe,
        'test_sortino': test_sortino,
        'test_mdd': test_mdd,
        'test_calmar': test_calmar,
        'test_return': test_return,
        'avg_weights': {**{t: float(avg_weights[i]) for i, t in enumerate(TICKERS)},
                        'CASH': float(avg_weights[5])},
        'n_eval_days': len(test_returns),
    }
    with open(results_dir / 'v2_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {results_dir / 'v2_results.json'}")

    # ========== V2 Baseline: PPO trained on full train, finetune test 20%, eval test 80% ==========
    print(f"\n{'='*60}")
    print("V2 Baseline: Pure PPO (train → finetune test 20% → eval test 80%)")
    print(f"{'='*60}")

    baseline_agent = PPOAgent(
        state_dim=70, hidden_dim=hidden_dim,
        actor_lr=ppo_cfg.get('actor_lr', 3e-4),
        critic_lr=ppo_cfg.get('critic_lr', 3e-4),
        gamma=ppo_cfg.get('gamma', 0.99),
        gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
        clip_epsilon=ppo_cfg.get('clip_epsilon', 0.2),
        entropy_coef=ppo_cfg.get('entropy_coef', 0.01),
        epochs_per_update=ppo_cfg.get('epochs_per_update', 10),
        batch_size=ppo_cfg.get('batch_size', 64),
        use_twin_critic=ppo_cfg.get('use_twin_critic', True),
        value_clip_epsilon=ppo_cfg.get('value_clip_epsilon', 0.2),
        dropout_rate=ppo_cfg.get('dropout_rate', 0.1),
        max_grad_norm=ppo_cfg.get('max_grad_norm', 0.5),
        critic_weight_decay=ppo_cfg.get('critic_weight_decay', 1e-5),
    )

    stock_features = [
        {'indicator': 'RSI', 'params': {'window': 14}},
        {'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
        {'indicator': 'Momentum', 'params': {'window': 10}},
        {'indicator': 'Bollinger', 'params': {'window': 20}},
        {'indicator': 'ATR', 'params': {'window': 14}},
    ]
    bl_revise_fn = build_revise_state(stock_features) if not hasattr(build_revise_state, '__module__') else None

    from feature_library import build_revise_state as _build_revise_state
    bl_revise_fn = _build_revise_state(stock_features)
    bl_port_fn = build_portfolio_features([
        {'indicator': 'momentum_rank', 'params': {'window': 20}},
        {'indicator': 'portfolio_volatility', 'params': {'window': 20}},
    ])

    # Train on full train period
    bl_train_env = PortfolioEnv(
        data_path, config,
        revise_state_fn=bl_revise_fn,
        portfolio_features_fn=bl_port_fn,
        detect_regime_fn=detect_market_regime,
        train_period=train_period,
        transaction_cost=transaction_cost,
    )
    bl_state_dim = bl_train_env.state_dim
    print(f"Baseline state dim: {bl_state_dim}")

    baseline_agent_bl = PPOAgent(
        state_dim=bl_state_dim, hidden_dim=hidden_dim,
        actor_lr=ppo_cfg.get('actor_lr', 3e-4),
        critic_lr=ppo_cfg.get('critic_lr', 3e-4),
        gamma=ppo_cfg.get('gamma', 0.99),
        gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
        clip_epsilon=ppo_cfg.get('clip_epsilon', 0.2),
        entropy_coef=ppo_cfg.get('entropy_coef', 0.01),
        epochs_per_update=ppo_cfg.get('epochs_per_update', 10),
        batch_size=ppo_cfg.get('batch_size', 64),
        use_twin_critic=ppo_cfg.get('use_twin_critic', True),
        value_clip_epsilon=ppo_cfg.get('value_clip_epsilon', 0.2),
        dropout_rate=ppo_cfg.get('dropout_rate', 0.1),
        max_grad_norm=ppo_cfg.get('max_grad_norm', 0.5),
        critic_weight_decay=ppo_cfg.get('critic_weight_decay', 1e-5),
    )

    bl_rewards = []
    bl_returns = []
    for episode in range(max_episodes):
        state = bl_train_env.reset()
        episode_reward = 0
        episode_returns = []
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            weights, log_prob = baseline_agent_bl.select_action(state)
            next_state, reward, done, info = bl_train_env.step(weights)
            states.append(state)
            actions.append(weights)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            episode_reward += reward
            episode_returns.append(info.get('portfolio_return', 0))
            state = next_state
        if len(states) > 1:
            baseline_agent_bl.update(states, actions, log_probs, rewards, dones, state)
        bl_rewards.append(episode_reward)
        bl_returns.extend(episode_returns)
        if (episode + 1) % 10 == 0:
            print(f"  Baseline episode {episode+1}/{max_episodes}: avg_reward={np.mean(bl_rewards[-10:]):.4f}")

    # Finetune on test first 20%
    bl_ft_env = PortfolioEnv(
        data_path, config,
        revise_state_fn=bl_revise_fn,
        portfolio_features_fn=bl_port_fn,
        detect_regime_fn=detect_market_regime,
        train_period=finetune_period,
        transaction_cost=transaction_cost,
    )
    for episode in range(finetune_episodes):
        state = bl_ft_env.reset()
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            weights, log_prob = baseline_agent_bl.select_action(state)
            next_state, reward, done, info = bl_ft_env.step(weights)
            states.append(state)
            actions.append(weights)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            state = next_state
        if len(states) > 1:
            baseline_agent_bl.update(states, actions, log_probs, rewards, dones, state)
        if (episode + 1) % 5 == 0:
            print(f"  Baseline fine-tune episode {episode+1}/{finetune_episodes}")

    # Eval on test last 80%
    bl_eval_env = PortfolioEnv(
        data_path, config,
        revise_state_fn=bl_revise_fn,
        portfolio_features_fn=bl_port_fn,
        detect_regime_fn=detect_market_regime,
        train_period=eval_period,
        transaction_cost=transaction_cost,
    )
    state = bl_eval_env.reset()
    done = False
    bl_test_returns = []
    while not done:
        weights, _ = baseline_agent_bl.select_action(state, deterministic=True)
        next_state, reward, done, info = bl_eval_env.step(weights)
        bl_test_returns.append(info.get('portfolio_return', 0))
        state = next_state

    bl_sharpe = sharpe_ratio(bl_test_returns)
    bl_sortino = sortino_ratio(bl_test_returns)
    bl_mdd = max_drawdown(bl_test_returns)
    bl_calmar = calmar_ratio(bl_test_returns)
    bl_return = (bl_eval_env.portfolio_value - 1.0) * 100

    print(f"\nBaseline Eval: Sharpe={bl_sharpe:.3f}, Return={bl_return:.2f}%, MDD={bl_mdd:.2f}%")

    # ========== Final comparison ==========
    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON V2 (Eval: {eval_period[0]} ~ {eval_period[1]})")
    print(f"{'='*80}")
    print(f"{'Metric':<24} {'LESR->FT':>12} {'PPO(train->FT)':>16}")
    print(f"{'-'*24} {'-'*12} {'-'*16}")
    print(f"{'Total Return':<24} {test_return:>11.2f}% {bl_return:>15.2f}%")
    print(f"{'Sharpe Ratio':<24} {test_sharpe:>12.3f} {bl_sharpe:>16.3f}")
    print(f"{'Sortino Ratio':<24} {test_sortino:>12.3f} {bl_sortino:>16.3f}")
    print(f"{'Max Drawdown':<24} {test_mdd:>11.2f}% {bl_mdd:>15.2f}%")
    print(f"{'Calmar Ratio':<24} {test_calmar:>12.3f} {bl_calmar:>16.3f}")
    print(f"{'='*80}")

    result['baseline_sharpe'] = bl_sharpe
    result['baseline_sortino'] = bl_sortino
    result['baseline_mdd'] = bl_mdd
    result['baseline_calmar'] = bl_calmar
    result['baseline_return'] = bl_return

    with open(results_dir / 'v2_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--v1_dir', required=True, help='Path to V1 results directory')
    parser.add_argument('--name', default='v2_finetune')
    args = parser.parse_args()
    run_v2_finetune(args.config, args.v1_dir, args.name)
