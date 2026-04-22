"""
LLM Prompt Templates for Portfolio Optimization LESR

Three code-generation prompts:
1. init_prompt: First iteration - state description + building blocks + code example
2. cot_prompt: After training - code + IC analysis + market context
3. next_iteration_prompt: Subsequent iterations - history + suggestions

Plus reward_config_prompt (kept as JSON selection) and _extract_python_code.

LLM 提示词模板模块 —— LESR 方法中 LLM 交互的核心。

=== 对应论文方法 ===
LESR (LLM-Empowered State Representation) 的迭代优化流程分为三个阶段，
每个阶段对应一个提示词模板：

  阶段1 - 初始化（init_prompt）：
    论文中的 "Initialization" 步骤。
    向 LLM 提供状态布局描述、可用构建块函数列表、市场统计信息，
    要求 LLM 生成 revise_state 和 intrinsic_reward 两个 Python 函数。
    目的：让 LLM 从零开始设计特征工程代码。

  阶段2 - COT 反馈（cot_prompt）：
    论文中的 "Chain-of-Thought Feedback" 步骤。
    将训练结果（性能指标、IC 分析、SHAP 值、市场环境）反馈给 LLM，
    引导 LLM 分析成功/失败原因并给出改进建议。
    目的：通过 IC+SHAP 交叉分析指导 LLM 理解哪些特征有效、哪些需要改进。

  阶段3 - 迭代改进（next_iteration_prompt）：
    论文中的 "Iterative Refinement" 步骤。
    在 COT 反馈的基础上，加上历史迭代记录，让 LLM 生成改进版代码。
    目的：基于前几轮的经验生成更优的状态表示。

此外还有 reward_config_prompt（奖励配置提示词），采用 JSON 选择范式而非代码生成。

=== 提示词设计思路 ===
1. 明确任务描述：每个提示词开头说明投资组合优化任务的具体设置
2. 状态布局说明：详细解释 120 维状态的含义，确保 LLM 理解输入格式
3. PPO 约束提醒：强调特征维度限制（3-8 维），防止过拟合
4. 构建块函数清单：提供预定义函数，降低 LLM 出错概率
5. 代码输出格式：明确要求输出 revise_state + intrinsic_reward 两个函数
6. 渐进式信息：后续迭代加入历史记录和 COT 建议，实现知识积累

=== LLM 输出要求 ===
- init_prompt / next_iteration_prompt：输出 Python 代码（包含在 ```python``` 代码块中）
- cot_prompt：输出分析文本和建议
- reward_config_prompt：输出 JSON 格式的奖励规则选择
"""

import json
import re


def _fmt(val, fmt_str):
    """Safe format: apply format if numeric, else return string as-is.

    安全格式化：数值类型应用格式化，其他类型直接返回字符串。
    """
    if isinstance(val, (int, float)):
        return format(val, fmt_str)
    return str(val)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown wrapping).

    从 LLM 响应中提取 JSON。支持以下格式：
    1. 直接 JSON 文本
    2. ```json ... ``` 代码块包裹
    3. 最外层花括号匹配提取
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = -1
    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


def _extract_python_code(text: str) -> str:
    """Extract Python code from LLM response.

    Tries:
      1. Extract from ```python ... ``` blocks
      2. Look for 'import numpy' to last function end
      3. Return full text as fallback

    从 LLM 响应中提取 Python 代码。按优先级尝试：
    1. 从 ```python ... ``` 代码块中提取（取最长的块）
    2. 从 'import numpy' 开始截取到最后一个函数结尾
    3. 回退返回完整文本
    """
    pattern = r'```(?:python)?\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    idx = text.find('import numpy')
    if idx >= 0:
        return text[idx:].strip()

    idx = text.find('def revise_state')
    if idx >= 0:
        import_idx = text.rfind('import', 0, idx)
        if import_idx >= 0:
            line_start = text.rfind('\n', 0, import_idx) + 1
            return text[line_start:].strip()
        return text[idx:].strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Building blocks description for prompts
# 构建块函数描述 —— 嵌入到 init_prompt 和 next_iteration_prompt 中
#
# 设计思路：
# 提供 9 个预定义的计算函数，让 LLM 无需从零实现金融指标。
# 每个函数都明确了输入/输出格式和用途，降低 LLM 代码出错的概率。
# 关键约束：限制选择 2-4 个函数，保持特征维度在 3-8 维，防止 PPO 过拟合。
# ---------------------------------------------------------------------------

BUILDING_BLOCKS_DESC = """
Available computation functions (import from feature_library):

1. compute_relative_momentum(prices, window=20)
   Input: prices = 1D array of close prices (length >= window)
   Output: scalar, this stock's excess return vs window-average
   Use case: identify relatively outperforming stocks

2. compute_realized_volatility(returns, window=20)
   Input: returns = 1D array of daily returns
   Output: scalar, realized volatility
   Use case: measure individual stock risk

3. compute_downside_risk(returns, window=20)
   Input: returns = 1D array of daily returns
   Output: scalar, downside semi-deviation
   Use case: measure downside risk

4. compute_multi_horizon_momentum(prices, windows=[5, 10, 20])
   Input: prices = 1D array of close prices
   Output: array of 3 scalars, momentum at each horizon
   Use case: capture trend at multiple time scales

5. compute_zscore_price(prices, window=20)
   Input: prices = 1D array
   Output: scalar, z-score of current price vs N-day mean
   Use case: mean reversion signal

6. compute_mean_reversion_signal(prices, window=20)
   Input: prices = 1D array
   Output: scalar, mean reversion strength
   Use case: identify overextended prices

7. compute_turnover_ratio(volumes, window=20)
   Input: volumes = 1D array
   Output: scalar, current volume / average volume
   Use case: liquidity detection

IMPORTANT: Select only 2-4 building blocks to keep feature dimension low (3-8 extra dims).
Too many features cause overfitting in PPO's on-policy training regime.
Prefer complementary signals (e.g., one trend + one risk + one volume indicator).

CRITICAL — Defensive Awareness:
At least ONE of your selected features should help the agent detect and respond to market stress.
Good choices: realized_volatility, downside_risk, or zscore_price (mean reversion = crash protection).
Pure momentum-only features leave the agent blind to drawdowns. A balanced feature set includes
both offensive (trend/momentum) and defensive (volatility/downside) signals.

Note: compute_cross_sectional_rank and compute_beta are portfolio-level
functions — they require data from all stocks, not available inside revise_state(s).
They are computed at the environment level, not inside your code.
"""


# ---------------------------------------------------------------------------
# 状态布局描述 —— 解释 120 维状态的含义
#
# 设计思路：
# 详细说明状态向量的布局（20天 × 6通道 × 单只股票），
# 并强调 revise_state 只处理单只股票（而非整个组合）。
# 这避免了 LLM 在代码中错误地遍历多只股票。
# ---------------------------------------------------------------------------
STATE_LAYOUT_DESC = """
The current state for EACH INDIVIDUAL STOCK is represented by a 120-dimensional Python NumPy array, denoted as `s`.

**CRITICAL: `s` contains data for ONE stock only. The environment calls revise_state(s) separately for each of the 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ). Do NOT loop over multiple assets inside revise_state.**

The 120 dimensions represent 20 trading days × 6 price channels for this single stock:
- `s[0]` through `s[5]`: [close, open, high, low, volume, adjusted_close] for day 1
- `s[6]` through `s[11]`: same 6 channels for day 2
- ...
- `s[114]` through `s[119]`: same 6 channels for day 20

In other words:
- `s[0::6]` = close prices (20 values, oldest to newest) for this one stock
- `s[1::6]` = open prices
- `s[2::6]` = high prices
- `s[3::6]` = low prices
- `s[4::6]` = volumes
- `s[5::6]` = adjusted close prices

Your revise_state function should compute features for THIS ONE stock only. Cross-stock comparisons (ranking, relative momentum) are handled at the portfolio level, not inside revise_state.
"""


REWARD_RULES = {
    'penalize_concentration': 'Penalty when any stock weight exceeds max_weight. Default max_weight=0.35, penalty=0.1.',
    'reward_diversification': 'Bonus when holding >= min_stocks above 5%. Default min_stocks=3, bonus=0.05.',
    'penalize_turnover': 'Penalty when turnover > threshold. Default threshold=0.1, penalty=0.15.',
    'regime_defensive': 'Bonus for holding cash when risk_level is high. Default crisis_threshold=0.6, cash_bonus=0.1.',
    'momentum_alignment': 'Bonus when weights correlate with momentum rank. Default bonus=0.05.',
    'volatility_scaling': 'Scale down reward in high-vol regime. Default vol_threshold=0.5, scale=0.5.',
    'drawdown_penalty': 'Penalty when drawdown exceeds threshold. Default dd_threshold=0.1, penalty=0.15.',
}


# ---------------------------------------------------------------------------
# 奖励规则字典 —— 用于 reward_config_prompt 中向 LLM 展示可用规则
# 每条规则包含名称和简要说明，LLM 从中选择 2-4 条并设置参数
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PPO 状态设计指导 —— 嵌入到提示词中约束 LLM 的特征设计
#
# 设计思路：
# 明确告知 LLM 关于 PPO 的约束：
# - PPO 是 on-policy 算法，训练数据有限，高维状态容易过拟合
# - 特征维度必须保持在 3-8 维
# - 应选择互补信号（趋势+风险+量价）而非冗余信号
# - intrinsic_reward 应鼓励探索信息丰富的状态
# ---------------------------------------------------------------------------
PPO_STATE_GUIDANCE = """
PPO-Specific Design Guidelines:
- The RL agent uses PPO (Proximal Policy Optimization) with Dirichlet distribution
  for portfolio weights. PPO is ON-POLICY: it learns from limited trajectory data.
- Feature dimension MUST be kept LOW (3-8 extra dimensions per stock).
  High-dimensional states cause overfitting because the policy memorizes
  training-specific patterns rather than learning generalizable features.
- Select only 2-4 building block functions. Prioritize COMPLEMENTARY signals
  (e.g., one trend indicator + one risk indicator + one volume indicator).
- intrinsic_reward should encourage exploration of informative states,
  not just amplify existing reward signals. Consider rewarding feature diversity
  or penalizing feature stagnation.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# 提示词构建函数
# ---------------------------------------------------------------------------

def build_init_prompt(market_stats: str) -> str:
    """Build initial code-generation prompt for first LESR iteration.

    构建初始化提示词 —— 对应论文 LESR 方法的 "Initialization" 阶段。

    【设计思路】
    这是第一次迭代时发送给 LLM 的提示词。目标是让 LLM 从零开始设计
    revise_state（状态修订）和 intrinsic_reward（内在奖励）两个函数。

    【提示词结构】
    1. 任务描述：说明投资组合优化的具体设置（5股票+现金）
    2. STATE_LAYOUT_DESC：详细解释 120 维状态的含义和布局
    3. PPO_STATE_GUIDANCE：PPO 算法约束和设计建议
    4. BUILDING_BLOCKS_DESC：可用的构建块函数清单
    5. Market Statistics：当前市场的统计信息
    6. intrinsic_reward 设计指导：要求使用原始维度+额外维度的组合
    7. 代码输出格式示例

    【LLM 输出要求】
    LLM 应返回包含两个函数的 Python 代码：
    - revise_state(s)：接受 120 维状态，返回扩展后的状态（120 + K 维）
    - intrinsic_reward(updated_s)：接受扩展状态，返回标量奖励值

    Args:
        market_stats: 由 market_stats.py 生成的市场统计文本
    """
    return f"""Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ)
      plus a CASH asset (6 assets total). The goal is to maximize risk-adjusted returns.
=========================================================

{STATE_LAYOUT_DESC}

{PPO_STATE_GUIDANCE}

{BUILDING_BLOCKS_DESC}

Market Statistics:
{market_stats}

Besides, we want you to design an intrinsic reward function based on the revise_state python function.

That is to say, we will:
1. use your revise_state python function to get an updated state: updated_s = revise_state(s)
2. use your intrinsic reward function to get an intrinsic reward: r = intrinsic_reward(updated_s)
3. to better design the intrinsic_reward, we recommend you use some source dim in the updated_s,
   which is between updated_s[0] and updated_s[119]
4. however, you must use the extra dim in your given revise_state python function,
   which is between updated_s[120] and the end of updated_s

Your task is to create two Python functions, named `revise_state`, which takes the current state `s`
as input and returns an updated state representation, and named `intrinsic_reward`, which takes the
updated state `updated_s` as input and returns an intrinsic reward. The functions should be executable
and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative
example of the expected output:

```python
import numpy as np
def revise_state(s):
    # Your state revision implementation goes here
    return updated_s
def intrinsic_reward(updated_s):
    # Your intrinsic reward code implementation goes here
    return intrinsic_reward
```"""


def build_cot_prompt(sample_results_text: str,
                     market_period_summary: str = "",
                     training_diagnostics: str = "") -> str:
    """Build COT feedback prompt after training.

    构建 COT（Chain-of-Thought）反馈提示词 —— 对应论文的 "COT Feedback" 阶段。

    【设计思路】
    在 PPO 训练完成后，将训练结果反馈给 LLM，引导其进行 Chain-of-Thought 分析。
    这不是让 LLM 直接生成代码，而是让 LLM 先分析成功/失败原因，再在下一轮
    next_iteration_prompt 中基于这些分析生成改进代码。

    【提示词结构】
    1. 背景说明：告知 LLM 已完成训练，并说明了监控的 4 个维度
       - 最终策略性能（累积奖励）
       - 每个特征维度的 IC（预测能力）
       - 每个特征维度的 SHAP 值（策略实际使用情况）
       - 市场环境和按状态分类的 IC 分析
    2. 训练结果：sample_results_text（由 build_ic_cot_prompt 生成）
    3. 市场环境：训练期间的市场状况描述
    4. 训练诊断：奖励趋势、Critic 损失等训练过程信息
    5. 分析提示：6 条分析建议（低性能分析、IC 分析、SHAP 分析等）

    【LLM 输出要求】
    LLM 应返回分析文本和改进建议，而非代码。

    Args:
        sample_results_text: 由 ic_analyzer.build_ic_cot_prompt() 生成的分析文本
        market_period_summary: 训练期间的市场环境描述
        training_diagnostics: 训练过程诊断信息（奖励趋势、损失变化）
    """
    diagnostics_section = ""
    if training_diagnostics:
        diagnostics_section = f"""
[Training Process Diagnostics]
{training_diagnostics}

Use these diagnostics to detect overfitting (e.g., critic loss not converging,
reward curve plateauing early, policy entropy collapsing).
"""
    return f"""We have successfully trained Reinforcement Learning (RL) policy using different
state revision codes and intrinsic reward function codes sampled by you, and each pair of code
is associated with the training of a policy relatively.

Throughout every state revision code's training process, we monitored:
1. The final policy performance (accumulated reward).
2. Every state revise dim's Information Coefficient (IC) with forward returns.
   IC measures predictive power: how well each dimension correlates with future returns.
3. Every state revise dim's SHAP value from the trained Critic network.
   SHAP reveals what the RL policy ACTUALLY uses: high SHAP = the critic relies on this dim.
   Ideal features have both high |IC| and high SHAP. Misused features (high SHAP, low IC)
   indicate overfitting. Underused features (high IC, low SHAP) need better representation.
4. Market environment context and regime-specific IC analysis.

Here are the results:
{sample_results_text}

[Market Environment During Training]
{market_period_summary}
{diagnostics_section}
You should analyze the results mentioned above and give suggestions about how to improve the
performance of the "state revision code".

Here are some tips for how to analyze the results:
(a) if you find a state revision code's performance is very low, then you should analyze to
    figure out why it fails
(b) if you find some dims' IC are more related to the final performance, you should analyze
    to figure out what makes it successful
(c) pay attention to SHAP values — features with high SHAP but low IC are being misused by
    the policy (overfitting); features with high IC but low SHAP are being ignored
(d) pay attention to regime-specific IC - features that work in trending markets may fail in
    volatile markets
(e) analyze how to improve both the "state revision code" and "intrinsic reward code"
(f) if feature dimension is too high (> 8), suggest reducing to avoid overfitting

Lets think step by step. Your solution should aim to improve the overall performance of the RL policy."""


def build_next_iteration_prompt(market_stats: str,
                                history_text: str,
                                cot_suggestions: str = "") -> str:
    """Build prompt for subsequent LESR iterations.

    构建迭代改进提示词 —— 对应论文的 "Iterative Refinement" 阶段。

    【设计思路】
    在第二轮及之后的迭代中使用。与 init_prompt 类似，但增加了：
    1. 历史迭代记录：前几轮尝试的代码及其性能
    2. COT 建议：上一轮训练后的分析和改进建议

    通过渐进式信息积累，让 LLM 基于前几轮的经验生成更优的状态表示。

    【提示词结构】
    1. 任务描述 + 状态布局 + PPO 约束 + 构建块（同 init_prompt）
    2. 更新的市场统计信息
    3. 历史迭代记录：前几轮的代码和性能
    4. COT 建议：上一轮的分析和改进建议
    5. 代码输出格式要求

    【LLM 输出要求】
    与 init_prompt 相同：返回包含 revise_state + intrinsic_reward 的 Python 代码。

    Args:
        market_stats: 更新的市场统计文本
        history_text: 格式化的历史迭代记录
        cot_suggestions: 上一轮的 COT 反馈建议
    """
    return f"""Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ) + CASH.
=========================================================

{STATE_LAYOUT_DESC}

{PPO_STATE_GUIDANCE}

{BUILDING_BLOCKS_DESC}

Updated Market Statistics:
{market_stats}

For this problem, we have some history experience for you, here are some state revision codes
we have tried in the former iterations:
{history_text}

{cot_suggestions}

Based on the former suggestions. We are seeking an improved state revision code and an improved
intrinsic reward code that can enhance the model's performance on the task.

Your task is to create two Python functions, named `revise_state`, which takes the current state `s`
as input and returns an updated state representation, and named `intrinsic_reward`, which takes the
updated state `updated_s` as input and returns an intrinsic reward. The functions should be executable
and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative
example of the expected output:

```python
import numpy as np
def revise_state(s):
    return updated_s
def intrinsic_reward(updated_s):
    return float_reward
```"""


def build_reward_config_prompt(market_stats: str, iteration: int,
                               history: list = None,
                               feature_rationale: str = "") -> str:
    """Build reward rule configuration prompt for LLM (unchanged JSON paradigm).

    构建奖励配置提示词 —— 采用 JSON 选择范式（非代码生成）。

    【设计思路】
    与上述三个代码生成提示词不同，这个提示词采用 JSON 选择范式：
    LLM 不需要生成代码，而是从预定义的 7 种奖励规则中选择 2-4 条并设置参数。
    这降低了 LLM 出错的概率，因为奖励规则的实现是固定的。

    【提示词结构】
    1. 基础奖励说明（Mean-Variance + intrinsic_reward）
    2. 市场统计信息
    3. 历史迭代结果（最近 3 轮）
    4. 特征选择说明
    5. 可用奖励规则列表（7 条，各含默认参数和说明）
    6. JSON 输出格式要求

    【LLM 输出要求】
    JSON 格式：
    {
      "reward_rules": [{"rule": "规则名", "params": {...}}, ...],
      "lambda": 0.5,
      "rationale": "选择理由"
    }

    Args:
        market_stats: 市场统计文本
        iteration: 当前迭代轮次
        history: 历史迭代结果列表
        feature_rationale: 当前轮次的特征选择说明
    """
    rules_cat = "\n".join(f"  - {k}: {v}" for k, v in REWARD_RULES.items())

    history_section = ""
    if history:
        history_section = "## Previous Iteration Results\n"
        for h in history[-3:]:
            history_section += f"### Iteration {h.get('iteration', '?')}\n"
            history_section += f"- Reward rules used: {h.get('reward_rules', 'N/A')}\n"
            history_section += f"- Sharpe: {_fmt(h.get('sharpe', 'N/A'), '.3f')}\n"
            history_section += f"- Max Drawdown: {_fmt(h.get('max_drawdown', 'N/A'), '.2f')}%\n"
            turnover = h.get('avg_turnover', 'N/A')
            turnover_str = f"{turnover:.3f}" if isinstance(turnover, (int, float)) else str(turnover)
            history_section += f"- Turnover: {turnover_str}\n"
            history_section += "\n"

    return f"""You are configuring the reward function for a PPO-based portfolio optimizer.

The base reward is Mean-Variance: r = portfolio_return - lambda * drawdown^2
Plus an intrinsic reward designed by another LLM code.

Your task: Select and parameterize additional reward rules to guide the agent.

## Market Statistics
{market_stats}

{history_section}

## Selected Features Rationale
{feature_rationale}

## Available Reward Rules
{rules_cat}

## Your Task
Select 2-4 reward rules and set their parameters. Consider:
1. Current market conditions (volatility, correlation structure)
2. Common failure modes (concentration, excessive turnover, ignoring regime)
3. Complementarity between rules (avoid redundant penalties)

## Output Format (JSON only)
```json
{{
  "reward_rules": [
    {{"rule": "penalize_concentration", "params": {{"max_weight": 0.35}}}},
    {{"rule": "regime_defensive", "params": {{"crisis_threshold": 0.6}}}}
  ],
  "lambda": 0.5,
  "rationale": "Brief explanation of why these rules were selected"
}}
```

Rules:
- Select 2-4 reward rules
- lambda controls risk aversion in base reward (0.1=aggressive, 1.0=conservative)
- Use default params unless market conditions suggest otherwise
"""
