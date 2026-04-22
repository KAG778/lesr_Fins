"""
LESR Controller for Portfolio Optimization

Main iteration loop (refactored to code-generation approach):
  1. LLM generates revise_state + intrinsic_reward Python code
  2. Code sandbox validates code (AST + test execution)
  3. Train PPO with validated code functions + reward rules
  4. Evaluate on validation set
  5. IC-based COT feedback for next iteration

LESR 控制器模块 —— 投资组合优化的主迭代循环。

采用代码生成（code-generation）范式（而非 JSON 选择范式）：
  步骤1: LLM 生成 revise_state + intrinsic_reward Python 代码
  步骤2: 代码沙箱验证（AST 白名单 + 测试执行 + 维度检测）
  步骤3: 用验证通过的代码训练 PPO（多样本并行）
  步骤4: 计算每个特征维度的 IC 和 SHAP 值
  步骤5: 生成 IC+COT 反馈，用于下一轮迭代
  步骤6: 最终在测试集上评估并与纯 PPO 基线对比

对应论文的完整 LESR 迭代优化流程。
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from prompts import (
    build_init_prompt, build_cot_prompt, build_next_iteration_prompt,
    build_reward_config_prompt, _extract_json, _extract_python_code,
)
from code_sandbox import validate as sandbox_validate
from ic_analyzer import compute_ic_profile, compute_regime_specific_ic, compute_critic_shap, build_ic_cot_prompt
from portfolio_features import PORTFOLIO_INDICATOR_REGISTRY, build_portfolio_features
from reward_rules import REWARD_RULE_REGISTRY, build_reward_rules
from regime_detector import detect_market_regime
from market_stats import get_market_stats, compute_strategy_hint
from portfolio_env import PortfolioEnv
from ppo_agent import PPOAgent
from metrics import sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio


TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']


def _fmt(val, fmt):
    """Format a value, handling non-numeric gracefully."""
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


class LESRController:
    """Orchestrates the LESR iteration loop for portfolio optimization.

    LESR 控制器：编排完整的迭代优化循环。
    管理 LLM 调用、代码验证、PPO 训练、IC/SHAP 分析和结果保存。
    """

    def __init__(self, config: dict, experiment_dir: str = 'results/default'):
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # LLM config
        llm_cfg = config.get('llm', {})
        self.client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY', llm_cfg.get('api_key', '')),
            base_url=llm_cfg.get('base_url', 'https://api.openai.com/v1'),
        )
        self.model = llm_cfg.get('model', 'gpt-4o-mini')
        self.temperature = llm_cfg.get('temperature', 0.7)

        # Experiment config
        exp_cfg = config.get('experiment', {})
        self.max_iterations = exp_cfg.get('max_iterations', 5)
        self.sample_count = exp_cfg.get('sample_count', 3)
        self.no_llm = exp_cfg.get('no_llm', False)

        # PPO config
        ppo_cfg = config.get('ppo', {})
        self.ppo_config = ppo_cfg
        self.max_episodes = ppo_cfg.get('max_episodes', 50)

        # Portfolio config
        port_cfg = config.get('portfolio', {})
        self.transaction_cost = port_cfg.get('transaction_cost', 0.001)
        self.default_lambda = port_cfg.get('default_lambda', 0.5)

        # Data paths
        data_cfg = config.get('data', {})
        self.data_path = data_cfg.get('pickle_file', 'data/portfolio_5stocks.pkl')

        # Train/val/test periods
        train_period = exp_cfg.get('train_period', ['2018-01-01', '2021-12-31'])
        val_period = exp_cfg.get('val_period', ['2022-01-01', '2022-12-31'])
        test_period = exp_cfg.get('test_period', ['2023-01-01', '2023-12-31'])
        self.train_period = tuple(train_period)
        self.val_period = tuple(val_period)
        self.test_period = tuple(test_period)

        # State tracking
        self.iteration_history = []
        self.best_sharpe = -float('inf')
        self.best_config = None

    def _call_llm(self, prompt: str, system_msg: str = None) -> str:
        """Call LLM with retry.

        调用 LLM API，带 3 次重试机制。默认使用量化组合管理员的系统提示词。
        """
        if system_msg is None:
            system_msg = "You are a quantitative portfolio manager. Always respond with valid JSON."
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"  LLM call attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
        raise RuntimeError("LLM call failed after 3 attempts")

    def _generate_code(self, iteration: int, n_samples: int = None) -> List[Dict]:
        """Step 1: LLM generates code, sandbox validates. Returns list of valid samples.

        步骤1：LLM 代码生成 + 沙箱验证。
        - 第1轮使用 build_init_prompt
        - 后续轮次使用 build_next_iteration_prompt（含历史记录和 COT 反馈）
        - 每次生成 n_samples 个代码样本，通过沙箱验证后保留
        - 若无有效样本，回退到默认代码
        """
        if n_samples is None:
            n_samples = self.sample_count
        print(f"\n[Iteration {iteration}] Step 1: Code Generation ({n_samples} samples)")

        env_tmp = PortfolioEnv(
            self.data_path, self.config,
            train_period=self.train_period,
            transaction_cost=self.transaction_cost,
        )
        training_states, _ = env_tmp.get_training_states(n_samples=200)
        market_stats = get_market_stats(training_states)

        if iteration == 1:
            prompt = build_init_prompt(market_stats)
        else:
            history_text = self._format_history()
            cot_suggestions = self.last_cot_feedback if hasattr(self, 'last_cot_feedback') else ""
            prompt = build_next_iteration_prompt(market_stats, history_text, cot_suggestions)

        valid_samples = []
        for sample_idx in range(n_samples):
            print(f"  Sampling code {sample_idx + 1}/{n_samples}...")
            try:
                response = self._call_llm(
                    prompt,
                    system_msg="You are a quantitative portfolio manager. Write Python code for revise_state and intrinsic_reward functions.")
                code = _extract_python_code(response)
                result = sandbox_validate(code)

                if result['ok']:
                    print(f"    Valid code: feature_dim={result['feature_dim']}, "
                          f"state_dim={result['state_dim']}")
                    valid_samples.append({
                        'code': code,
                        'revise_state_fn': result['revise_state'],
                        'intrinsic_reward_fn': result['intrinsic_reward'],
                        'feature_dim': result['feature_dim'],
                        'state_dim': result['state_dim'],
                    })
                else:
                    print(f"    Invalid code: {result['errors'][:2]}")
            except Exception as e:
                print(f"    Sample {sample_idx + 1} failed: {e}")

        if not valid_samples:
            print("  No valid code samples, using default")
            valid_samples = [self._default_code_config()]

        return valid_samples

    def _configure_rewards(self, iteration: int, feature_rationale: str = "") -> dict:
        """Step 2: LLM configures reward rules (JSON selection, unchanged).

        步骤2：LLM 配置奖励规则（JSON 选择范式）。
        LLM 从 7 种预定义奖励规则中选择 2-4 条并设置参数。
        """
        print(f"\n[Iteration {iteration}] Step 2: Reward Configuration")

        env_tmp = PortfolioEnv(
            self.data_path, self.config,
            train_period=self.train_period,
            transaction_cost=self.transaction_cost,
        )
        training_states, _ = env_tmp.get_training_states(n_samples=100)
        market_stats = get_market_stats(training_states)
        strategy_hint = compute_strategy_hint(training_states)

        prompt = build_reward_config_prompt(
            market_stats=market_stats,
            iteration=iteration,
            history=self.iteration_history,
            feature_rationale=feature_rationale,
            strategy_hint=strategy_hint,
        )

        response = self._call_llm(prompt)

        try:
            parsed = _extract_json(response)
        except ValueError as e:
            print(f"  JSON parse error: {e}")
            return self._default_reward_config()

        rules_list = parsed.get('reward_rules', [])
        lam = parsed.get('lambda', self.default_lambda)
        rationale = parsed.get('rationale', '')

        valid_rules = []
        for rule in rules_list:
            name = rule.get('rule', '')
            if name in REWARD_RULE_REGISTRY:
                entry = REWARD_RULE_REGISTRY[name]
                merged = dict(entry['default_params'])
                merged.update(rule.get('params', {}))
                for pk, pv in merged.items():
                    if pk in entry['param_ranges']:
                        lo, hi = entry['param_ranges'][pk]
                        merged[pk] = type(pv)(np.clip(pv, lo, hi))
                valid_rules.append({'rule': name, 'params': merged})

        reward_fn = build_reward_rules(valid_rules) if valid_rules else None

        print(f"  Selected: {len(valid_rules)} reward rules, lambda={lam:.2f}")

        return {
            'reward_rules': valid_rules,
            'reward_rules_fn': reward_fn,
            'lambda': lam,
            'rationale': rationale,
        }

    def _train_ppo(self, code_sample: dict, reward_config: dict,
                   portfolio_features_fn=None) -> dict:
        """Step 3: Train PPO agent with code-generated functions.

        步骤3：使用 LLM 生成的代码函数训练 PPO 智能体。
        训练完成后自动计算：
        - IC profile：每个额外维度的预测能力
        - SHAP profile：策略实际使用了哪些维度
        - 训练诊断：奖励趋势、损失变化
        """
        print("\nStep 3: PPO Training")

        env = PortfolioEnv(
            self.data_path, self.config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=portfolio_features_fn,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            train_period=self.train_period,
            transaction_cost=self.transaction_cost,
        )

        state_dim = env.state_dim
        print(f"  State dim: {state_dim}")

        hidden_dim = self.ppo_config.get('hidden_dim', 256)
        agent = PPOAgent(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            actor_lr=self.ppo_config.get('actor_lr', 3e-4),
            critic_lr=self.ppo_config.get('critic_lr', 3e-4),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_epsilon=self.ppo_config.get('clip_epsilon', 0.2),
            entropy_coef=self.ppo_config.get('entropy_coef', 0.01),
            epochs_per_update=self.ppo_config.get('epochs_per_update', 10),
            batch_size=self.ppo_config.get('batch_size', 64),
            use_twin_critic=self.ppo_config.get('use_twin_critic', True),
            value_clip_epsilon=self.ppo_config.get('value_clip_epsilon', 0.2),
            dropout_rate=self.ppo_config.get('dropout_rate', 0.1),
            max_grad_norm=self.ppo_config.get('max_grad_norm', 0.5),
            critic_weight_decay=self.ppo_config.get('critic_weight_decay', 1e-5),
        )

        all_rewards = []
        all_returns = []
        all_actor_losses = []
        all_critic_losses = []
        reward_first_half = []
        reward_second_half = []

        for episode in range(self.max_episodes):
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
                update_info = agent.update(states, actions, log_probs, rewards, dones, state)
                all_actor_losses.append(update_info['actor_loss'])
                all_critic_losses.append(update_info['critic_loss'])

            all_rewards.append(episode_reward)
            all_returns.extend(episode_returns)

            mid = self.max_episodes // 2
            if episode < mid:
                reward_first_half.append(episode_reward)
            else:
                reward_second_half.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_rew = np.mean(all_rewards[-10:])
                sharpe = sharpe_ratio(all_returns[-252:]) if len(all_returns) > 10 else 0.0
                print(f"  Episode {episode+1}/{self.max_episodes}: "
                      f"avg_reward={avg_rew:.4f}, sharpe={sharpe:.3f}")

        final_sharpe = sharpe_ratio(all_returns)
        final_sortino = sortino_ratio(all_returns)
        final_mdd = max_drawdown(all_returns)
        final_calmar = calmar_ratio(all_returns)
        total_return = (env.portfolio_value - 1.0) * 100

        print(f"  Training complete: Sharpe={final_sharpe:.3f}, "
              f"MDD={final_mdd:.2f}%, Return={total_return:.2f}%")

        # Compute IC profile and Critic SHAP
        ic_profile = {}
        shap_profile = {}
        regime_ic = {}
        ir_stats = {}
        try:
            revised_states, forward_returns, regime_labels = env.get_revised_states(300)
            if len(revised_states) > 20:
                ic_profile = compute_ic_profile(revised_states, forward_returns)
                regime_ic = compute_regime_specific_ic(revised_states, forward_returns, regime_labels)

                # Critic SHAP: what does the trained policy actually use?
                device = str(agent.device)
                n_env_samples = min(100, len(env.dates) - 22)
                env_state_indices = np.linspace(20, len(env.dates) - 2,
                                                n_env_samples, dtype=int)
                env_states = np.array([env._compute_state(idx) for idx in env_state_indices])
                # Extra dims start at index 50 (after compressed raw per stock)
                shap_profile = compute_critic_shap(
                    agent.critic, env_states, extra_start=50, device=device)
                if shap_profile:
                    print(f"  SHAP computed for {len(shap_profile)} extra dims")

                if code_sample.get('intrinsic_reward_fn') and len(revised_states) > 10:
                    ir_values = [code_sample['intrinsic_reward_fn'](s) for s in revised_states[:50]]
                    ir_mean = float(np.mean(ir_values))
                    ir_corr = float(np.corrcoef(ir_values, forward_returns[:50])[0, 1]) if len(ir_values) > 5 else 0.0
                    ir_stats = {'mean': ir_mean, 'correlation_with_performance': ir_corr if not np.isnan(ir_corr) else 0.0}
        except Exception as e:
            print(f"  IC/SHAP computation skipped: {e}")

        # Build training diagnostics for COT feedback
        training_diagnostics = ""
        if all_rewards:
            trend = "improving" if (np.mean(reward_second_half) > np.mean(reward_first_half)) else "declining"
            training_diagnostics += f"Reward trend: {trend} "
            training_diagnostics += f"(first half avg={np.mean(reward_first_half):.4f}, "
            training_diagnostics += f"second half avg={np.mean(reward_second_half):.4f})\n"
        if all_critic_losses:
            training_diagnostics += f"Critic loss: initial={all_critic_losses[0]:.4f}, "
            training_diagnostics += f"final={all_critic_losses[-1]:.4f}\n"
            if all_critic_losses[-1] > all_critic_losses[0]:
                training_diagnostics += "  -> Critic loss INCREASING: possible overfitting\n"
        if all_actor_losses:
            training_diagnostics += f"Actor loss: initial={all_actor_losses[0]:.4f}, "
            training_diagnostics += f"final={all_actor_losses[-1]:.4f}\n"

        return {
            'agent': agent,
            'env': env,
            'sharpe': final_sharpe,
            'sortino': final_sortino,
            'max_drawdown': final_mdd,
            'calmar': final_calmar,
            'total_return': total_return,
            'all_returns': all_returns,
            'portfolio_value': env.portfolio_value,
            'ic_profile': ic_profile,
            'shap_profile': shap_profile,
            'regime_ic': regime_ic,
            'intrinsic_reward_stats': ir_stats,
            'training_diagnostics': training_diagnostics,
        }

    def _evaluate(self, agent: PPOAgent, code_sample: dict,
                  reward_config: dict, period: tuple = None,
                  label: str = "Validation") -> dict:
        """Evaluate agent on a given period (val or test).

        在指定时间段（验证集或测试集）上评估智能体。
        使用确定性策略（取 Dirichlet 分布均值），记录各项绩效指标和平均权重。
        """
        if period is None:
            period = self.test_period

        print(f"\n  [{label} Evaluation] ({period[0]} ~ {period[1]})")

        env = PortfolioEnv(
            self.data_path, self.config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=None,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            train_period=period,
            transaction_cost=self.transaction_cost,
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

        print(f"    Sharpe={ep_sharpe:.3f}, Sortino={ep_sortino:.3f}, "
              f"MDD={ep_mdd:.2f}%, Return={ep_return:.2f}%")
        print(f"    Avg weights: " +
              ", ".join(f"{TICKERS[i]}={avg_weights[i]:.3f}" for i in range(5)) +
              f", CASH={avg_weights[5]:.3f}")

        return {
            f'{label.lower()}_sharpe': ep_sharpe,
            f'{label.lower()}_sortino': ep_sortino,
            f'{label.lower()}_max_drawdown': ep_mdd,
            f'{label.lower()}_calmar': ep_calmar,
            f'{label.lower()}_total_return': ep_return,
            f'{label.lower()}_returns': returns,
            f'{label.lower()}_avg_weights': {**{TICKERS[i]: float(avg_weights[i]) for i in range(5)},
                                              'CASH': float(avg_weights[5])},
        }

    def _default_code_config(self) -> dict:
        """Fallback code config when LLM fails.

        LLM 代码生成失败时的默认回退配置。
        使用相对动量 + 已实现波动率两个构建块。
        """
        code = """import numpy as np
from feature_library import compute_relative_momentum, compute_realized_volatility
def revise_state(s):
    closes = s[0::6]
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    mom = compute_relative_momentum(closes, 20)
    vol = compute_realized_volatility(returns, 20)
    return np.concatenate([s, [mom, vol]])
def intrinsic_reward(updated_s):
    return 0.01 * abs(updated_s[120]) / (updated_s[121] + 0.01)
"""
        result = sandbox_validate(code)
        if result['ok']:
            return {
                'code': code,
                'revise_state_fn': result['revise_state'],
                'intrinsic_reward_fn': result['intrinsic_reward'],
                'feature_dim': result['feature_dim'],
                'state_dim': result['state_dim'],
            }
        return {
            'code': 'import numpy as np\ndef revise_state(s): return s\ndef intrinsic_reward(s): return 0.0',
            'revise_state_fn': lambda s: s,
            'intrinsic_reward_fn': lambda s: 0.0,
            'feature_dim': 0,
            'state_dim': 120,
        }

    def _default_reward_config(self) -> dict:
        """Fallback reward config."""
        rules = [
            {'rule': 'penalize_concentration', 'params': {'max_weight': 0.35, 'penalty': 0.1}},
            {'rule': 'penalize_turnover', 'params': {'threshold': 0.1, 'penalty': 0.15}},
        ]
        return {
            'reward_rules': rules,
            'reward_rules_fn': build_reward_rules(rules),
            'lambda': self.default_lambda,
            'rationale': 'Default fallback rules',
        }

    def _format_history(self) -> str:
        """Format iteration history for prompt.

        将迭代历史格式化为文本，嵌入到 next_iteration_prompt 中。
        只展示最近 3 轮的记录，避免提示词过长。
        """
        lines = []
        for h in self.iteration_history[-3:]:
            lines.append(f"Iteration {h.get('iteration', '?')}:")
            lines.append(f"  Train Sharpe: {_fmt(h.get('sharpe', 'N/A'), '.3f')}")
            lines.append(f"  Return: {_fmt(h.get('total_return', 'N/A'), '.2f')}%")
            if h.get('ic_profile'):
                lines.append(f"  IC profile: {h['ic_profile']}")
            lines.append("")
        return "\n".join(lines)

    def _get_market_summary(self) -> str:
        """Get brief market summary for COT prompt.

        获取简要的市场环境描述，注入到 COT 反馈中。
        包含年化收益率、年化波动率和趋势判断。
        """
        try:
            env_tmp = PortfolioEnv(
                self.data_path, self.config,
                train_period=self.train_period,
                transaction_cost=self.transaction_cost,
            )
            training_states, forward_returns = env_tmp.get_training_states(n_samples=100)
            if len(forward_returns) > 10:
                avg_ret = float(np.mean(forward_returns)) * 252 * 100
                vol = float(np.std(forward_returns)) * np.sqrt(252) * 100
                trend = "bullish" if avg_ret > 5 else ("bearish" if avg_ret < -5 else "neutral")
                return (f"Training period: {self.train_period[0]} ~ {self.train_period[1]}\n"
                        f"  Annualized return: {avg_ret:.1f}%\n"
                        f"  Annualized volatility: {vol:.1f}%\n"
                        f"  Trend: {trend}")
        except Exception:
            pass
        return f"Training period: {self.train_period[0]} ~ {self.train_period[1]}"

    def run(self):
        """Run the full LESR iteration loop with multi-sample code generation.

        运行完整的 LESR 迭代循环。
        每轮迭代：代码生成 → 奖励配置 → 多样本训练 → COT 反馈 → 保存结果。
        全部迭代完成后在测试集上评估，并与纯 PPO 基线对比。
        """
        print("=" * 60)
        print("LESR Portfolio Optimization (Code-Generation Mode)")
        print(f"Iterations: {self.max_iterations}, Samples: {self.sample_count}, "
              f"PPO episodes: {self.max_episodes}")
        print(f"Train: {self.train_period}, Val: {self.val_period}, Test: {self.test_period}")
        if self.no_llm:
            print("Mode: NO_LLM (using default features & reward rules)")
        print("=" * 60)

        self.last_cot_feedback = ""

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*60}")

            try:
                # First iteration uses more samples for diverse initialization
                n_samples = 3 if iteration == 1 else self.sample_count
                if self.no_llm:
                    code_samples = [self._default_code_config()]
                    reward_config = self._default_reward_config()
                else:
                    code_samples = self._generate_code(iteration, n_samples=n_samples)
                    reward_config = self._configure_rewards(
                        iteration, "Code-generated features")

                sample_results = []
                for s_idx, code_sample in enumerate(code_samples):
                    print(f"\n  --- Training Sample {s_idx + 1}/{len(code_samples)} ---")
                    train_result = self._train_ppo(code_sample, reward_config)

                    sample_results.append({
                        'code': code_sample.get('code', ''),
                        'train_result': train_result,
                        'performance': {
                            'sharpe': train_result['sharpe'],
                            'total_return': train_result['total_return'],
                            'max_drawdown': train_result['max_drawdown'],
                        },
                        'ic_profile': train_result.get('ic_profile', {}),
                        'shap_profile': train_result.get('shap_profile', {}),
                        'regime_ic': train_result.get('regime_ic', {}),
                        'intrinsic_reward_stats': train_result.get('intrinsic_reward_stats', {}),
                    })

                best_idx = max(range(len(sample_results)),
                              key=lambda i: sample_results[i]['performance']['sharpe'])
                best = sample_results[best_idx]
                best_perf = best['performance']
                print(f'  Best sample: {best_idx + 1} '
                      f'(Train Sharpe={best_perf["sharpe"]:.3f}, '
                      f'Return={best_perf["total_return"]:.2f}%)')

                if not self.no_llm and len(sample_results) > 0:
                    best_train_result = sample_results[best_idx].get("train_result", {})
                    cot_text = build_ic_cot_prompt(
                        sample_results, best_idx,
                        market_period_summary=self._get_market_summary(),
                        training_diagnostics=best_train_result.get("training_diagnostics", ""),
                    )
                    self.last_cot_feedback = cot_text
                    print(f"  COT feedback generated ({len(cot_text)} chars)")

                record = {
                    "iteration": iteration,
                    "n_samples": len(code_samples),
                    "best_sample_idx": best_idx,
                    "reward_rules": [r["rule"] for r in reward_config.get("reward_rules", [])],
                    "lambda": reward_config.get("lambda", self.default_lambda),
                    "sharpe": best["performance"]["sharpe"],
                    "max_drawdown": best["performance"]["max_drawdown"],
                    "total_return": best["performance"]["total_return"],
                    "ic_profile": {str(k): f"{v:.4f}" for k, v in best.get("ic_profile", {}).items()},
                }
                self.iteration_history.append(record)

                self._save_iteration(iteration, code_samples[best_idx], reward_config,
                                     best["train_result"], record)

                if best["performance"]["sharpe"] > self.best_sharpe:
                    self.best_sharpe = best["performance"]["sharpe"]
                    self.best_config = {
                        "iteration": iteration,
                        "code": code_samples[best_idx].get("code", ""),
                        "reward_config": reward_config,
                    }
                    model_path = self.experiment_dir / "best_model.pt"
                    best["train_result"]["agent"].save(str(model_path))
                    print(f"  New best model! Train Sharpe={self.best_sharpe:.3f}")


            except Exception as e:
                print(f"  Iteration {iteration} failed: {e}")
                import traceback
                traceback.print_exc()

        self._save_summary()

        # Final test evaluation + baseline comparison
        test_result = {}
        baseline_result = {}
        if self.best_config:
            # Load best model and evaluate on test set
            print(f"\n{'='*60}")
            print("STEP 5: Final Test Evaluation + Baseline Comparison")
            print(f"{'='*60}")

            # Recreate code_sample from best config
            best_code = self.best_config.get('code', '')
            code_sample = self._default_code_config()
            if best_code:
                try:
                    from code_sandbox import validate
                    r = validate(best_code)
                    if r['ok']:
                        code_sample = {
                            'code': best_code,
                            'revise_state_fn': r['revise_state'],
                            'intrinsic_reward_fn': r['intrinsic_reward'],
                            'feature_dim': r['feature_dim'],
                            'state_dim': r['state_dim'],
                        }
                except Exception:
                    pass

            reward_config = self.best_config.get('reward_config', self._default_reward_config())

            # Create env to get correct state_dim, then load model
            env_check = PortfolioEnv(
                self.data_path, self.config,
                revise_state_fn=code_sample.get('revise_state_fn'),
                detect_regime_fn=detect_market_regime,
                intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
                train_period=self.test_period,
                transaction_cost=self.transaction_cost,
            )
            best_agent = PPOAgent(
                state_dim=env_check.state_dim,
                hidden_dim=self.ppo_config.get('hidden_dim', 256),
                use_twin_critic=self.ppo_config.get('use_twin_critic', True),
            )
            model_path = self.experiment_dir / 'best_model.pt'
            best_agent.load(str(model_path))

            test_result = self._evaluate(
                best_agent, code_sample, reward_config,
                period=self.test_period, label="Test")

            # Run pure PPO baseline for comparison
            baseline_result = self._run_baseline_comparison()

            # Print comparison table
            self._print_comparison(test_result, baseline_result)

            # Save test + baseline results
            summary_extra = {
                'test_result': {k: v for k, v in test_result.items() if 'returns' not in k},
                'baseline_result': {k: v for k, v in baseline_result.items() if 'returns' not in k},
            }
            with open(self.experiment_dir / 'final_comparison.json', 'w') as f:
                json.dump(summary_extra, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"LESR Complete. Best Train Sharpe: {self.best_sharpe:.3f}")

    def _save_iteration(self, iteration, code_sample, reward_config,
                        train_result, record):
        """Save iteration results to disk."""
        iter_dir = self.experiment_dir / f'iteration_{iteration}'
        iter_dir.mkdir(exist_ok=True)

        with open(iter_dir / 'code.py', 'w') as f:
            f.write(code_sample.get('code', ''))

        save_cfg = {
            'reward_rules': reward_config.get('reward_rules', []),
            'lambda': reward_config.get('lambda', self.default_lambda),
            'feature_dim': code_sample.get('feature_dim', 0),
            'state_dim': code_sample.get('state_dim', 120),
        }
        with open(iter_dir / 'config.json', 'w') as f:
            json.dump(save_cfg, f, indent=2, default=str)

        with open(iter_dir / 'metrics.json', 'w') as f:
            json.dump(record, f, indent=2, default=str)

        train_result['agent'].save(str(iter_dir / 'model.pt'))

    def _run_baseline_comparison(self) -> dict:
        """Train pure PPO (no LLM, no LESR) for comparison.

        训练纯 PPO 基线（无 LLM、无 LESR 状态修订）用于对比。
        使用原始压缩状态，相同超参数，评估 LESR 增益。
        """
        print(f"\n  --- Pure PPO Baseline (No LLM / No LESR) ---")

        # Use no revise_state — raw compressed state only
        env = PortfolioEnv(
            self.data_path, self.config,
            detect_regime_fn=detect_market_regime,
            train_period=self.train_period,
            transaction_cost=self.transaction_cost,
        )

        state_dim = env.state_dim
        print(f"    Baseline state dim: {state_dim}")

        agent = PPOAgent(
            state_dim=state_dim,
            hidden_dim=self.ppo_config.get('hidden_dim', 256),
            actor_lr=self.ppo_config.get('actor_lr', 3e-4),
            critic_lr=self.ppo_config.get('critic_lr', 3e-4),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_epsilon=self.ppo_config.get('clip_epsilon', 0.2),
            entropy_coef=self.ppo_config.get('entropy_coef', 0.01),
            epochs_per_update=self.ppo_config.get('epochs_per_update', 10),
            batch_size=self.ppo_config.get('batch_size', 64),
            use_twin_critic=self.ppo_config.get('use_twin_critic', True),
            value_clip_epsilon=self.ppo_config.get('value_clip_epsilon', 0.2),
            dropout_rate=self.ppo_config.get('dropout_rate', 0.1),
            max_grad_norm=self.ppo_config.get('max_grad_norm', 0.5),
            critic_weight_decay=self.ppo_config.get('critic_weight_decay', 1e-5),
        )

        for episode in range(self.max_episodes):
            state = env.reset()
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
                state = next_state
            if len(states) > 1:
                agent.update(states, actions, log_probs, rewards, dones, state)

        # Evaluate baseline on test set
        code_sample_none = {'revise_state_fn': None, 'intrinsic_reward_fn': None}
        reward_cfg_none = {'reward_rules_fn': None}
        result = self._evaluate(
            agent, code_sample_none, reward_cfg_none,
            period=self.test_period, label="Baseline")
        return result

    def _print_comparison(self, test_result: dict, baseline_result: dict):
        """Print side-by-side comparison table.

        打印 LESR+PPO vs 纯 PPO 的对比表格。
        包含收益率、夏普比率、索提诺比率、最大回撤、卡尔马比率和平均权重。
        """
        # Normalize keys: both results use {label}_sharpe format
        # test_result keys: test_sharpe, test_sortino, ...
        # baseline_result keys: baseline_sharpe, baseline_sortino, ...
        def _get(result, suffix, default=float('nan')):
            for prefix in ['test', 'baseline', 'val']:
                v = result.get(f'{prefix}_{suffix}')
                if v is not None:
                    return v
            return default

        print(f"\n{'='*60}")
        print(f"FINAL COMPARISON (Test: {self.test_period[0]} ~ {self.test_period[1]})")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'LESR+PPO':>12} {'Pure PPO':>12}")
        print(f"{'-'*20} {'-'*12} {'-'*12}")

        metrics = [
            ('Total Return', 'total_return', '%.2f%%'),
            ('Sharpe Ratio', 'sharpe', '%.3f'),
            ('Sortino Ratio', 'sortino', '%.3f'),
            ('Max Drawdown', 'max_drawdown', '%.2f%%'),
            ('Calmar Ratio', 'calmar', '%.3f'),
        ]

        for label, suffix, fmt in metrics:
            lesr_val = _get(test_result, suffix)
            base_val = _get(baseline_result, suffix)
            lesr_str = fmt % lesr_val if not np.isnan(lesr_val) else 'N/A'
            base_str = fmt % base_val if not np.isnan(base_val) else 'N/A'
            marker = " *" if not np.isnan(lesr_val) and not np.isnan(base_val) and lesr_val > base_val else ""
            print(f"{label:<20} {lesr_str:>12} {base_str:>12}{marker}")

        # Avg weights comparison
        lesr_w = _get(test_result, 'avg_weights', {})
        base_w = _get(baseline_result, 'avg_weights', {})
        print(f"\n{'Avg Weights':<20} {'LESR+PPO':>12} {'Pure PPO':>12}")
        print(f"{'-'*20} {'-'*12} {'-'*12}")
        for name in ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ', 'CASH']:
            lw = f"{lesr_w.get(name, 0):.3f}" if lesr_w else 'N/A'
            bw = f"{base_w.get(name, 0):.3f}" if base_w else 'N/A'
            print(f"  {name:<18} {lw:>12} {bw:>12}")
        print(f"{'='*60}")
        print("  (* = LESR+PPO wins on this metric)")

    def _save_summary(self):
        """Save overall experiment summary."""
        summary = {
            'best_train_sharpe': self.best_sharpe,
            'best_iteration': self.best_config['iteration'] if self.best_config else None,
            'iterations': len(self.iteration_history),
            'history': self.iteration_history,
        }
        with open(self.experiment_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
