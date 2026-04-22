"""
V2 Baseline: Pure PPO, train on full train_period,
continue training on first 20% of test_period, evaluate on last 80%.
Mirrors V2 (fine-tune) evaluation protocol.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'core'))

import yaml
from feature_library import build_revise_state
from portfolio_features import build_portfolio_features
from regime_detector import detect_market_regime
from portfolio_env import PortfolioEnv
from ppo_agent import PPOAgent
from metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']


def run_baseline_v2(config_path: str, experiment_name: str = 'baseline_v2'):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results_dir = PROJECT_DIR / 'results' / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = config.get('experiment', {})
    train_period = tuple(exp_cfg.get('train_period', ['2018-01-01', '2021-12-31']))
    test_period = tuple(exp_cfg.get('test_period', ['2023-01-01', '2023-12-31']))

    ppo_cfg = config.get('ppo', {})
    max_episodes = ppo_cfg.get('max_episodes', 50)
    hidden_dim = ppo_cfg.get('hidden_dim', 256)
    transaction_cost = config.get('portfolio', {}).get('transaction_cost', 0.001)
    data_path = config.get('data', {}).get('pickle_file', 'data/portfolio_5stocks.pkl')

    stock_features = [
        {'indicator': 'RSI', 'params': {'window': 14}},
        {'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
        {'indicator': 'Momentum', 'params': {'window': 10}},
        {'indicator': 'Bollinger', 'params': {'window': 20}},
        {'indicator': 'ATR', 'params': {'window': 14}},
    ]
    portfolio_features = [
        {'indicator': 'momentum_rank', 'params': {'window': 20}},
        {'indicator': 'portfolio_volatility', 'params': {'window': 20}},
    ]
    revise_fn = build_revise_state(stock_features)
    port_fn = build_portfolio_features(portfolio_features)

    # Split test_period: first 20% for fine-tune, last 80% for evaluation
    import pickle
    with open(data_path, 'rb') as f:
        price_data = pickle.load(f)

    all_dates = sorted(price_data.keys())
    test_dates = [d for d in all_dates if test_period[0] <= d <= test_period[1]]
    split_idx = int(len(test_dates) * 0.2)
    finetune_period = (test_dates[0], test_dates[split_idx])
    eval_period = (test_dates[split_idx + 1], test_dates[-1])

    print("=" * 60)
    print("V2 Baseline: Pure PPO (train → finetune test 20% → eval test 80%)")
    print(f"Train period:      {train_period}")
    print(f"Fine-tune (20%):   {finetune_period} ({split_idx + 1} days)")
    print(f"Eval (80%):        {eval_period} ({len(test_dates) - split_idx - 1} days)")
    print("=" * 60)

    # Step 1: Train on full train_period
    env = PortfolioEnv(
        data_path, config,
        revise_state_fn=revise_fn,
        portfolio_features_fn=port_fn,
        detect_regime_fn=detect_market_regime,
        train_period=train_period,
        transaction_cost=transaction_cost,
    )
    state_dim = env.state_dim
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

    all_rewards = []
    all_returns = []
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_returns = []
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            weights, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(weights)
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
        if (episode + 1) % 10 == 0:
            avg_rew = np.mean(all_rewards[-10:])
            sharpe = sharpe_ratio(all_returns[-252:]) if len(all_returns) > 10 else 0.0
            print(f"  Episode {episode+1}/{max_episodes}: avg_reward={avg_rew:.4f}, sharpe={sharpe:.3f}")

    train_sharpe = sharpe_ratio(all_returns)
    train_mdd = max_drawdown(all_returns)
    train_return = (env.portfolio_value - 1.0) * 100
    print(f"\nTraining: Sharpe={train_sharpe:.3f}, MDD={train_mdd:.2f}%, Return={train_return:.2f}%")

    # Step 2: Fine-tune on first 20% of test_period
    finetune_env = PortfolioEnv(
        data_path, config,
        revise_state_fn=revise_fn,
        portfolio_features_fn=port_fn,
        detect_regime_fn=detect_market_regime,
        train_period=finetune_period,
        transaction_cost=transaction_cost,
    )

    finetune_episodes = min(15, max_episodes // 3)
    print(f"\n  --- Fine-tuning on test period ({finetune_episodes} episodes) ---")
    for episode in range(finetune_episodes):
        state = finetune_env.reset()
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
            state = next_state
        if len(states) > 1:
            agent.update(states, actions, log_probs, rewards, dones, state)
        if (episode + 1) % 5 == 0:
            print(f"    Fine-tune episode {episode+1}/{finetune_episodes}")

    # Step 3: Evaluate on last 80% of test_period
    eval_env = PortfolioEnv(
        data_path, config,
        revise_state_fn=revise_fn,
        portfolio_features_fn=port_fn,
        detect_regime_fn=detect_market_regime,
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
    print(f"Eval Results ({eval_period[0]} to {eval_period[1]})")
    print(f"{'='*60}")
    print(f"  Total Return:  {test_return:.2f}%")
    print(f"  Sharpe Ratio:  {test_sharpe:.3f}")
    print(f"  Sortino Ratio: {test_sortino:.3f}")
    print(f"  Max Drawdown:  {test_mdd:.2f}%")
    print(f"  Calmar Ratio:  {test_calmar:.3f}")

    result = {
        'method': 'V2_baseline',
        'train_period': list(train_period),
        'finetune_period': finetune_period,
        'eval_period': eval_period,
        'train_sharpe': train_sharpe, 'train_mdd': train_mdd, 'train_return': train_return,
        'test_sharpe': test_sharpe, 'test_sortino': test_sortino,
        'test_mdd': test_mdd, 'test_calmar': test_calmar, 'test_return': test_return,
        'avg_weights': {**{t: float(avg_weights[i]) for i, t in enumerate(TICKERS)},
                        'CASH': float(avg_weights[5])},
        'n_eval_days': len(test_returns),
    }
    with open(results_dir / 'baseline_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {results_dir / 'baseline_results.json'}")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', default='baseline_v2')
    args = parser.parse_args()
    run_baseline_v2(args.config, args.name)
