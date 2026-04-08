# llm_rl_trading_finsaber 项目完整分析

> **LESR 从机器人控制到金融 DRL 决策的迁移实践**
>
> 分析日期：2026-04-02
> 项目版本：finsaber_native composite collab full
> 分析重点：Prompt 工程、代码生成机制、迭代优化、与原始 LESR 对比、改进建议

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构分析](#2-系统架构分析)
3. [Prompt 工程细节](#3-prompt-工程细节)
4. [代码采样和验证机制](#4-代码采样和验证机制)
5. [迭代优化工作流](#5-迭代优化工作流)
6. [与原始 LESR 的对比](#6-与原始-lesr-的对比)
7. [当前不足分析](#7-当前不足分析)
8. [改进建议](#8-改进建议)

---

## 1. 项目概述

### 1.1 项目定位

`llm_rl_trading_finsaber` 是 **LESR (LLM-Empowered State Representation for RL)** 在金融交易领域的应用实践，将原始 LESR 从机器人控制任务迁移到金融深度强化学习（DRL）决策场景。

### 1.2 核心创新

```
┌─────────────────────────────────────────────────────────────┐
│           LESR in Financial Trading DRL                     │
├─────────────────────────────────────────────────────────────┤
│  🎯 目标：利用 LLM 自动设计交易状态表示和内在奖励函数        │
│                                                              │
│  🔧 核心组件：                                                │
│  • revise_state(s)    - 状态表示增强（原始状态 → 扩展特征）  │
│  • intrinsic_reward(s) - 内在奖励设计（引导探索/风险控制）  │
│                                                              │
│  📊 评估体系：                                                │
│  • G0: Baseline (无 LESR)                                   │
│  • G1: Revise Only (仅状态增强)                              │
│  • G2: Intrinsic Only (仅内在奖励)                          │
│  • G3: Joint (状态增强 + 内在奖励)                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 项目结构

```
llm_rl_trading_finsaber/
├── src/
│   ├── lesr/                    # LESR 核心模块
│   │   ├── llm_sampler.py       # LLM 采样器
│   │   ├── prompt_templates.py  # Prompt 模板引擎
│   │   └── revision_candidates.py # 静态候选库
│   ├── llm/
│   │   └── deepseek_client.py   # DeepSeek API 客户端
│   ├── pipeline/
│   │   ├── demo.py              # 主流程控制
│   │   └── branch_iteration_worker.py # 算法分支迭代
│   ├── drl/
│   │   └── finsaber_native_runner.py # Native DRL 运行器
│   └── finsaber_native/         # FINSABER 原生实现
│       ├── env_stocktrading.py  # 交易环境
│       └── state_contract.py    # 状态契约定义
├── configs/
│   └── current_baseline/        # 当前最佳配置
├── scripts/
│   └── run.py                   # 统一入口
└── docs/steps/                  # 迁移文档
```

---

## 2. 系统架构分析

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        llm_rl_trading_finsaber 架构                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  数据源 (Data)    │
│  • 价格数据       │ ────┐
│  • 技术指标       │     │
│  • 市场特征       │     │
└──────────────────┘     │
                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    Stage 1: LESR 初始化                                   │
├───────────────────────────────────────────────────────────────────────────┤
│  1.1 状态契约定义 (Native State Contract)                                 │
│      • cash + close_prices + holdings + indicators                       │
│      • 不同于 generic OHLCV schema                                         │
│                                                                           │
│  1.2 Prompt 构建                                                          │
│      • system_prompt: trading_lesr_prior_v1                              │
│      • user_prompt: task + state_desc + constraints                      │
│      • state_contract_note: native-specific warnings                      │
│                                                                           │
│  1.3 LLM 采样                                                            │
│      • DeepSeek API                                                      │
│      • k=3 候选/轮                                                        │
│      • temperature=0.2 (确定性)                                          │
└───────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    Stage 2: 代码验证与训练                                │
├───────────────────────────────────────────────────────────────────────────┤
│  2.1 代码解析                                                             │
│      • extract_code(): 提取 Python 代码块                                 │
│      • is_valid_code(): 检查必需函数                                      │
│                                                                           │
│  2.2 功能验证                                                             │
│      • revise_state() 维度检查                                            │
│      • intrinsic_reward() 范围检查 [-100, 100]                            │
│      • 数值稳定性检查 (NaN/Inf)                                           │
│                                                                           │
│  2.3 并行训练 (4 Workers)                                                │
│      • G0: Baseline                                                      │
│      • G1: Revise Only                                                   │
│      • G2: Intrinsic Only                                                │
│      • G3: Joint                                                         │
│                                                                           │
│  支持 4 种 DRL 算法: A2C, PPO, SAC, TD3                                   │
└───────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    Stage 3: 评估与反馈                                    │
├───────────────────────────────────────────────────────────────────────────┤
│  3.1 性能评估                                                             │
│      • Sharpe Ratio                                                      │
│      • Cumulative Returns                                                │
│      • Maximum Drawdown                                                  │
│                                                                           │
│  3.2 Lipschitz 常数计算                                                  │
│      • 每个状态维度与奖励的连续性                                         │
│      • 识别关键特征维度                                                   │
│                                                                           │
│  3.3 COT 反馈生成                                                         │
│      • 对比最佳/最差候选                                                 │
│      • 分析成功/失败模式                                                 │
│      • 提出改进建议                                                       │
└───────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    Stage 4: 迭代优化 (10 轮)                             │
├───────────────────────────────────────────────────────────────────────────┤
│  4.1 更新 Prompt                                                          │
│      • 融合历史反馈                                                       │
│      • 压缩历史经验 (最多 3 轮)                                           │
│      • 明确避免重复                                                       │
│                                                                           │
│  4.2 场景族 (Scenario Family) 采样                                       │
│      • trend_follow: 趋势跟踪                                            │
│      • mean_revert: 均值回归                                             │
│      • risk_shield: 风险屏蔽                                             │
│                                                                           │
│  4.3 算法分支独立迭代                                                     │
│      • 每个算法维护独立的候选历史                                        │
│      • per_algorithm_branches 模式                                       │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LESR 金融交易数据流                                 │
└─────────────────────────────────────────────────────────────────────────────┘

原始市场数据 (OHLCV)
    │
    ▼
┌──────────────────┐
│  FeatureEngineer │ → 技术指标 (SMA, RSI, Volatility)
└────────┬─────────┘
         │
         ▼
Native Raw State
┌─────────────────────────────────────────────────────────────────┐
│ [cash, close_1..N, holdings_1..N, indicators_1..N×M]            │
│                                                                  │
│ 例如: [100000, 150.2, 148.5, ..., 0, 50, ..., SMA, RSI, ...]  │
└────────┬────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     ▼
    revise_state(s)                    ┌──────────────┐
         │                             │  G0/G2 Mode  │
         │                             │  (Raw State) │
         │                             └──────┬───────┘
         ▼                                    │
    Extended State                          │
┌───────────────────────────────────┐        │
│ [raw_state | added_features]     │        │
│                                   │        │
│ 例如: [..., momentum, exposure,   │        │
│        concentration, regime, ... │        │
└────────┬──────────────────────────┘        │
         │                                    │
         ├──────────────┐                     │
         │              │                     │
         ▼              ▼                     │
    ┌─────────┐   ┌──────────┐               │
    │ G1 Mode │   │ G3 Mode  │               │
    │ (Policy)│   │ (Policy) │               │
    └────┬────┘   └────┬─────┘               │
         │              │                     │
         └──────┬───────┴─────────────────────┘
                │
                ▼
         ┌──────────────┐
         │ Actor Network │ → 动作 (连续权重)
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  Trading Env │ → 执行交易
         └──────┬───────┘
                │
                ▼
         ┌──────────────────────────────────────┐
         │  r_total = r_env + w_int × r_int     │
         │                                       │
         │  • r_env: 环境收益 (PnL)              │
         │  • r_int: intrinsic_reward(state)    │
         │  • w_int: intrinsic_w (默认 0.02)    │
         └──────────────────────────────────────┘
                │
                ▼
         Replay Buffer → Critic Network 训练
                │
                ▼
    Episode End → Lipschitz 常数计算 → LLM 反馈
```

### 2.3 状态契约对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    三种状态契约对比                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Generic LESR Schema (原始机器人控制)                                       │
├───────────────────────────────────────────────────────────────────────────┤
│ 维度公式: len(global) + len(assets) × (6 + len(indicators))              │
│                                                                           │
│ 字段顺序:                                                                 │
│ • global_features (2 dims):                                              │
│   - cash                                                                 │
│   - portfolio_value                                                      │
│                                                                           │
│ • per_asset (6 + len(indicators) dims):                                  │
│   - open                                                                 │
│   - high                                                                 │
│   - low                                                                  │
│   - close                                                                │
│   - volume                                                               │
│   - holding                                                              │
│   - indicators (SMA, RSI, ...)                                           │
│                                                                           │
│ 例如: [100000, 105000, 150.2, 151.0, 148.5, 150.0, 1e6, 0, 50, SMA, ...] │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ FINSABER Native Contract (当前项目)                                       │
├───────────────────────────────────────────────────────────────────────────┤
│ 维度公式: 1 + 2×stock_dim + len(indicators)×stock_dim                    │
│                                                                           │
│ 字段顺序:                                                                 │
│ • cash (1 dim):                                                          │
│   - 当前现金                                                              │
│                                                                           │
│ • close_prices (stock_dim dims):                                         │
│   - 每个资产的收盘价                                                      │
│                                                                           │
│ • holdings (stock_dim dims):                                             │
│   - 每个资产的持仓量                                                      │
│                                                                           │
│ • indicators (stock_dim × len(indicators) dims):                         │
│   - 每个资产的技术指标                                                    │
│   - 按 indicator-major 排列 (不同与 generic 的 asset-major)               │
│                                                                           │
│ 例如 (5 资产, 3 指标):                                                    │
│ [100000,                                                                │
│  150.2, 148.5, 152.0, 149.8, 151.2,  # close prices (5)                 │
│  0, 50, 0, 30, 20,               # holdings (5)                         │
│  SMA_1, SMA_2, ..., SMA_5,       # SMA indicators (5)                   │
│  RSI_1, RSI_2, ..., RSI_5,       # RSI indicators (5)                   │
│  VOL_1, VOL_2, ..., VOL_5]       # Volatility indicators (5)            │
│                                                                           │
│ 总计: 1 + 5 + 5 + 15 = 26 dims                                           │
└───────────────────────────────────────────────────────────────────────────┘

⚠️  关键差异:
1. Native 没有 OHLCV 中的 open/high/low/volume
2. Native 的 indicators 按 indicator-major 排列 (不是 asset-major)
3. Native 的 close 和 holdings 是独立的 block
4. Prompt 必须明确说明这些差异，否则 LLM 会生成错误的索引
```

---

## 3. Prompt 工程细节

### 3.1 System Prompt 设计

#### 3.1.1 核心设计原则

```python
# 文件: src/lesr/prompt_templates.py -> build_system_prompt()

MODE = "trading_lesr_prior_v1"  # 金融交易专用模式

System Prompt 组成要素:
┌─────────────────────────────────────────────────────────────┐
│ 1. 角色设定                                                 │
│    "You are designing revise_state(s) and                  │
│     intrinsic_reward(updated_s) for LESR-style             │
│     trading RL."                                            │
├─────────────────────────────────────────────────────────────┤
│ 2. 目标说明                                                 │
│    "Produce code that improves sample efficiency and       │
│     risk-adjusted performance while preserving             │
│     policy-action sensitivity."                            │
├─────────────────────────────────────────────────────────────┤
│ 3. 输出契约                                                 │
│    • import numpy as np                                    │
│    • def revise_state(s): ...                              │
│    • def intrinsic_reward(updated_s): ...                  │
│    • No markdown, no explanations                          │
├─────────────────────────────────────────────────────────────┤
│ 4. 硬约束条件                                               │
│    • No future information leakage                         │
│    • NumPy-only operations                                 │
│    • Keep outputs finite and bounded                       │
│    • Intrinsic reward in [-100, 100]                       │
│    • Must work on raw state alone (G2 mode)                │
├─────────────────────────────────────────────────────────────┤
│ 5. 失败模式警告                                             │
│    • Action-insensitive intrinsic design                   │
│    • Large monotonic concentration push                    │
│    • Overly sparse trigger logic                           │
│    • Unstable sign-flip terms                              │
└─────────────────────────────────────────────────────────────┘
```

#### 3.1.2 金融交易先验知识 (Empirical Priors)

```python
# System Prompt 包含的实验先验 (disable_priors=False)

✅ Positive/Robust Families (成功模式):
1. action_sensitive_spread_rank_v4
   • spread-rank + risk-budget + bounded normalization
   • 在 A2C/SAC 上重复显示正向增量

2. action_sensitive_spread_rank_v6_penalty_clip & v7_balanced
   • 在 SAC/TD3 上改善方向性增益

3. action_sensitive_step42_bull_preserve_boundclip_v17
   • 在 index-like 协议中有用

✅ Mechanism Requirements (机制要求):
• TD3 特定要求:
  - concentration/bound penalties 必须平滑且状态依赖
  - 优先使用 confidence-gated penalties
  - 鼓励 action-relevant ranking terms
  - 避免"改变奖励总量但不改变动作行为"的设计
  - 优先 portfolio-footprint-aware terms

✅ Structural Motifs (结构模式):
• revise_state 应包含:
  - robust trend proxy
  - volatility proxy
  - concentration/risk-budget proxy
  - confidence proxy
  - portfolio-memory terms (cash ratio, exposure, concentration, entropy)
  - bounded transforms (tanh, safe ratio, normalized spread)

• intrinsic_reward 应包含:
  - positive term: action-relevant spread/rank/trend-confidence interaction
  - negative term: concentration/bound risk with confidence-aware clipping
  - risk-adjusted and turnover-aware shaping
  - final bounded aggregation in stable numeric range
```

#### 3.1.3 特征组语义 (Feature Groups)

```python
# Prompt 中定义的 4 类特征组

FEATURE_GROUPS = [
    "portfolio_memory",    # 投资组合记忆
    "regime",              # 市场状态
    "dispersion",          # 离散度
    "running_risk_state"   # 运行风险状态
]

┌─────────────────────────────────────────────────────────────┐
│ 特征组详细定义                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. portfolio_memory (投资组合记忆)                         │
│    • cash_ratio: 现金比例                                   │
│    • exposure: 敞口                                         │
│    • concentration: 集中度 (最大持仓比例)                   │
│    • entropy: 持仓熵 (分散度)                               │
│    • rebalancing_pressure: 再平衡压力                       │
│                                                             │
│ 2. regime (市场状态)                                       │
│    • volatility_level: 波动率水平                           │
│    • volatility_ratio: 波动率比值                           │
│    • drawdown: 回撤                                         │
│    • market_stress: 市场压力                                │
│    • trend_strength: 趋势强度                               │
│                                                             │
│ 3. dispersion (离散度)                                     │
│    • spread: 买卖价差                                       │
│    • rank: 相对排名                                         │
│    • breadth: 广度 (上涨股票占比)                           │
│    • cross_asset_disagreement: 跨资产分歧                   │
│    • winner_minus_loser: 赢家减输家                         │
│                                                             │
│ 4. running_risk_state (运行风险状态)                       │
│    • return_ema: 收益指数移动平均                           │
│    • return_squared_ema: 收益平方 EMA                       │
│    • drawdown_ema: 回撤 EMA                                 │
│    • turnover_ema: 换手率 EMA                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 User Prompt 设计

#### 3.2.1 初始 Prompt (Initial Prompt)

```python
# 文件: src/lesr/prompt_templates.py -> build_initial_prompt()

def build_initial_prompt(
    task_description: str,
    state_desc: List[str],
    state_contract_note: str = "",
) -> str:
    """
    构建初始 Prompt
    """
    # 1. 标准化状态描述
    state_desc = _normalize_state_desc(state_desc)

    # 2. 推断状态维度
    total_dim = _infer_state_dim_from_desc(state_desc)

    # 3. 格式化详细内容
    detail_content = "\n".join([f"- {d}" for d in state_desc])

    # 4. 添加状态契约说明
    extra_note = f"\nAuthoritative state contract note:\n{state_contract_note}\n" \
                 if state_contract_note else ""

    return f"""
Revise the state representation for a reinforcement learning agent.
=========================================================
The agent's task description is:
{task_description}
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}
{extra_note}

You should design a task-related state representation based on the source {total_dim} dims.
Use the details above to compute new features, then concatenate them to the original state.

Besides, we want you to design an intrinsic reward function based on the revise_state function.

That is to say, we will evaluate the intrinsic reward in two modes:
1. G2-like intrinsic-only mode: `r_int = intrinsic_reward(s)` using the original source state.
2. G3-like joint mode: `updated_s = revise_state(s)` then `r_int = intrinsic_reward(updated_s)`.
3. Therefore intrinsic_reward must be valid and informative on the original dims alone.
4. If revised extra dims are present, intrinsic_reward should use them to refine the signal rather than requiring them to exist.

Constraints:
- Do NOT use any future data.
- Only use NumPy operations.
- Keep outputs numeric and bounded.
- Do NOT trivially clip intrinsic reward to a tiny range like [-1, 1].
- Keep intrinsic reward informative and roughly comparable to environment reward scale,
  and bounded in [-100, 100].
- Intrinsic reward must have a raw-state fallback path when only the source dims are available.
- The primary intrinsic signal must come from the raw/source dims; revised extra dims may only
  refine, gate, or denoise that same signal.
- Reject near-constant or almost-zero intrinsic designs on raw states; the raw-state branch should
  remain non-trivial by itself.
- When revised extra dims are present, use them as additional context to improve the same signal
  instead of carrying the sole predictive content.
- Prefer revise_state features that expose portfolio memory, regime, and risk-budget context.
- Prefer intrinsic_reward designs that improve risk-adjusted behavior and avoid unstable
  portfolio-weight jumps.
- Feature-group semantics:
  - `portfolio_memory`: cash ratio, holdings, exposure, concentration, entropy, rebalancing pressure
  - `regime`: volatility level, volatility ratio, drawdown, market stress, trend-strength regime
  - `dispersion`: spread, rank, breadth, cross-asset disagreement, winner-minus-loser structure
  - `running_risk_state`: return EMA, return-squared EMA, drawdown EMA, turnover EMA
- Output ONLY code. No comments. No markdown.
- Do not use hard-coded out-of-range indices; all index access must be valid for the returned vector length.
- If possible, declare `FEATURE_GROUPS = [...]` before the functions to indicate which semantic
  feature groups the candidate is using.

Your task is to create executable `revise_state` and `intrinsic_reward` functions.
"""
```

#### 3.2.2 迭代 Prompt (Next Iteration Prompt)

```python
# 文件: src/lesr/prompt_templates.py -> build_next_iteration_prompt()

def build_next_iteration_prompt(
    task_description: str,
    state_desc: List[str],
    history_results: List[str],      # 历史结果摘要
    history_suggestions: List[str],  # 历史建议
    state_contract_note: str = "",
) -> str:
    """
    构建迭代 Prompt - 包含历史反馈
    """
    # 1. 压缩历史内容
    former_history = _format_history_block(
        history_results,
        history_suggestions
    )

    return f"""
Revise the state representation for a reinforcement learning agent.
=========================================================
The agent's task description is:
{task_description}
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}
{extra_note}

You should design a task-related state representation based on the source {total_dim} dims
to better for reinforcement training, using the detailed information mentioned above to do
some calculations, and feel free to do complex calculations, and then concatenate them
to the source state.

Recent compressed history from former iterations:
{former_history}

Based on the history above, seek an improved state revision code and an improved intrinsic
reward code. Do not repeat the same feature family, normalization, gating rule, or intrinsic
mechanism unless you are making a clear structural change.

That is to say, we will evaluate the intrinsic reward in two modes:
1. G2-like intrinsic-only mode: `r_int = intrinsic_reward(s)` using the original source state.
2. G3-like joint mode: `updated_s = revise_state(s)` then `r_int = intrinsic_reward(updated_s)`.
3. intrinsic_reward must therefore work on the original dims alone.
4. If revised extra dims are present, use them to enrich the same mechanism instead of
  making them a hard dependency.

Constraints:
- Do NOT use any future data.
- Only use NumPy operations.
- Keep outputs numeric and bounded.
- Any division or normalization must be division-safe: use a positive denominator floor
  and a fallback branch when the denominator is too small.
- Do not use raw ratios such as `x / mean(y)` or `x / std(y)` without an explicit
  denominator guard and a safe fallback value.
- Prefer numerically stable transforms: clipped z-score, bounded spread/rank,
  tanh-normalized proxy, or guarded difference-over-scale.
- Do NOT trivially clip intrinsic reward to a tiny range like [-1, 1].
- Keep intrinsic reward informative and roughly comparable to environment reward scale,
  and bounded in [-100, 100].
- Intrinsic reward must have a raw-state fallback path and should remain useful even
  without revised extra dims.
- The primary intrinsic signal must come from raw/source dims; revised extra dims may
  only refine, gate, or denoise that same signal.
- Reject near-constant or almost-zero intrinsic designs on raw states; the raw-state
  branch should stay non-trivial on its own.
- If revised extra dims are available, use them as extra evidence or gating, not as
  the sole source of signal.
- Prefer revise_state features that expose portfolio memory, regime, and risk-budget context.
- Prefer intrinsic_reward designs that improve risk-adjusted behavior and avoid unstable
  portfolio-weight jumps.
- Output ONLY code. No comments. No markdown.
- Prefer a materially new candidate over a small cosmetic rewrite of prior code.
- Mechanism diversity requirement: do not reuse the same revise_state scaffold with only
  renamed variables or minor coefficient changes. Change at least one of:
  1. the main feature family,
  2. the normalization strategy,
  3. the gating logic,
  4. the intrinsic decomposition into opportunity/risk terms.

Your task is to create executable `revise_state` and `intrinsic_reward` functions ready
for integration into the RL environment.
"""
```

#### 3.2.3 CoT Prompt (Chain-of-Thought 反馈)

```python
# 文件: src/lesr/prompt_templates.py -> build_cot_prompt()

def build_cot_prompt(
    codes: List[str],              # 候选代码列表
    scores: List[float],           # 性能分数
    max_id: int,                   # 最佳候选 ID
    factors: List[List[float]],    # Lipschitz 常数
    dims: List[int],               # 扩展后维度
    source_dim: int,               # 原始维度
    task_name: str = "Final Policy Performance",
) -> str:
    """
    构建 CoT 反馈 Prompt
    """
    # 1. 生成反馈文本
    s_feedback = ""
    for i, code in enumerate(codes):
        s_feedback += f"========== Code {i+1} ==========\n"
        s_feedback += code + "\n"
        s_feedback += f"Performance: {round(scores[i], 4)}\n"

        # 2. 分析原始维度的 Lipschitz 常数
        s_feedback += (
            f"Source state dims s[0] ~ s[{source_dim-1}] Lipschitz constants:\n"
        )
        for k in range(source_dim):
            s_feedback += f"  s[{k}] = {round(factors[i][k], 4)}\n"

        # 3. 分析扩展维度的 Lipschitz 常数
        extra_dims = max(dims[i] - source_dim, 0)
        if extra_dims > 0:
            s_feedback += f"Extra dims s[{source_dim}] ~ s[{dims[i]-1}]:\n"
            for k in range(source_dim, dims[i]):
                s_feedback += f"  s[{k}] = {round(factors[i][k], 4)}\n"

        s_feedback += "\n"

    return f"""
We have successfully trained Reinforcement Learning (RL) policy using {len(codes)}
different state revision codes and intrinsic reward function codes sampled by you,
and each pair of code is associated with the training of a policy.

Throughout every state revision code's training process, we monitored:
1. The final policy performance (accumulated reward).
2. Most importantly, every state revise dim's Lipschitz constant with the reward.
   That is to say, you can see which state revise dim is more related to the reward
   and which dim can contribute to enhancing the continuity of the reward function mapping.
   Lower Lipschitz constant means better continuity and smoother of the mapping.
   Note: Lower Lipschitz constant is better.

Here are the results:
{s_feedback}

You should analyze the results mentioned above and give suggestions about how to improve
the performance of the state revision code.

Here are some tips for how to analyze the results:
(a) if you find a state revision code's performance is very low, then you should analyze
    to figure out why it fails
(b) if you find some dims are more related to the final performance, then you should
    analyze what makes it succeed
(c) you should also analyze how to improve the performance of the state revision code
    and intrinsic reward code later

Lets think step by step. Your solution should aim to improve the overall performance
of the RL policy.
"""
```

### 3.3 Prompt 历史压缩机制

```python
# 文件: src/lesr/prompt_templates.py

# 历史压缩参数
_PROMPT_HISTORY_MAX_ITERS = 3          # 最多保留 3 轮历史
_PROMPT_HISTORY_MAX_RESULT_CHARS = 1200  # 结果摘要最多 1200 字符
_PROMPT_HISTORY_MAX_SUGGESTION_CHARS = 700  # 建议摘要最多 700 字符
_PROMPT_HISTORY_TOTAL_CHAR_BUDGET = 5000   # 总预算 5000 字符

def _format_history_block(
    history_results: List[str],
    history_suggestions: List[str]
) -> str:
    """
    压缩历史内容到 Prompt
    """
    pairs = list(zip(history_results, history_suggestions))

    # 1. 限制历史轮数
    if len(pairs) > _PROMPT_HISTORY_MAX_ITERS:
        omitted = len(pairs) - _PROMPT_HISTORY_MAX_ITERS
        pairs = pairs[-_PROMPT_HISTORY_MAX_ITERS:]
        header = f"\n[Only the latest {_PROMPT_HISTORY_MAX_ITERS} former iterations "
                 f"are shown; {omitted} earlier iterations omitted.]\n"
    else:
        header = ""

    # 2. 字符预算控制
    remaining_budget = _PROMPT_HISTORY_TOTAL_CHAR_BUDGET
    blocks: List[str] = [header] if header else []

    for offset, (result_text, suggestion_text) in enumerate(pairs):
        iter_idx = start_idx + offset

        # 3. 压缩单个块
        result_trimmed = _trim_block(
            _compact_history_text(result_text, max_lines=10),
            _PROMPT_HISTORY_MAX_RESULT_CHARS,
        )
        suggestion_trimmed = _trim_block(
            _compact_history_text(suggestion_text, max_lines=8),
            _PROMPT_HISTORY_MAX_SUGGESTION_CHARS,
        )

        block = (
            f"\n\nFormer Iteration:{iter_idx} Summary\n"
            f"{result_trimmed}"
            f"\n\nKeep/avoid guidance from Iteration:{iter_idx}\n"
            f"{suggestion_trimmed}"
        )

        # 4. 预算检查
        if remaining_budget <= 0:
            break
        if len(block) > remaining_budget:
            block = _trim_block(block, remaining_budget)
            blocks.append(block)
            remaining_budget = 0
            break

        blocks.append(block)
        remaining_budget -= len(block)

    return "".join(blocks).strip()
```

---

## 4. 代码采样和验证机制

### 4.1 LLM 采样流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM 采样完整流程                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Start
 │
 ├─→ 构建 Prompt
 │   • system_prompt: build_system_prompt()
 │   • user_prompt: build_initial_prompt() / build_next_iteration_prompt()
 │
 ├─→ 调用 DeepSeek API
 │   • model: deepseek-chat
 │   • temperature: 0.2 (确定性)
 │   • max_tokens: 3500
 │   • timeout: 90s
 │
 ├─→ 提取代码 (extract_code)
 │   ├─→ 正则匹配: r"```(?:python)?(.*?)```"
 │   ├─→ Fallback: 去除 ``` 标记
 │   └─→ 返回: 纯 Python 代码字符串
 │
 ├─→ 验证代码 (is_valid_code)
 │   ├─→ 检查: "def revise_state" in code
 │   ├─→ 检查: "def intrinsic_reward" in code
 │   └─→ 返回: Boolean
 │
 ├─→ 有效？
 │   ├─→ YES: 添加到 codes 列表
 │   └─→ NO: 重试 (最多 max_retries=4 次)
 │
 └─→ 返回: codes + raw_responses
```

### 4.2 代码提取算法

```python
# 文件: src/lesr/llm_sampler.py

import re

CODE_BLOCK_RE = re.compile(r"```(?:python)?(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code(text: str) -> str:
    """
    从 LLM 输出中提取 Python 代码
    """
    # 1. 尝试匹配代码块
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    # 2. Fallback: 去除 markdown 标记
    cleaned = text.replace("```python", "").replace("```", "")
    return cleaned.strip()


def extract_lesr_code(text: str) -> str:
    """
    提取 LESR 特定代码 (revise_state + intrinsic_reward)
    """
    code = extract_code(text)

    # 3. 检查必需函数
    if ("def revise_state" in code) and ("def intrinsic_reward" in code):
        return _trim_to_function_block(code)

    # 4. Fallback: 提取完整响应体
    return _trim_to_function_block(text)


def _trim_to_function_block(text: str) -> str:
    """
    裁剪到函数块
    """
    cleaned = text.replace("\r\n", "\n").replace("`", "")

    # 查找起始位置
    starts = [
        cleaned.find("import numpy as np"),
        cleaned.find("def revise_state"),
        cleaned.find("def intrinsic_reward"),
    ]
    starts = [idx for idx in starts if idx >= 0]
    start = min(starts) if starts else 0

    return cleaned[start:].strip() + "\n"
```

### 4.3 代码验证机制

```python
# 文件: src/lesr/llm_sampler.py

def is_valid_code(code: str) -> bool:
    """
    快速验证代码是否包含必需函数
    """
    return ("def revise_state" in code) and ("def intrinsic_reward" in code)


def sample_candidates(
    client: DeepSeekClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    k: int,                    # 采样数量
    temperature: float,
    max_tokens: int,
    max_retries: int,          # 最大重试次数
) -> Tuple[List[str], List[Dict]]:
    """
    采样多个候选代码
    """
    codes: List[str] = []
    raw_responses: List[Dict] = []

    for i in range(k):
        retries = 0
        while retries <= max_retries:
            # 1. 构建 messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # 2. 调用 LLM
            content = client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw_responses.append({"index": i, "content": content})

            # 3. 提取并验证代码
            code = extract_code(content)
            if is_valid_code(code):
                codes.append(code)
                break

            # 4. 失败重试
            retries += 1

        # 5. 超过最大重试次数
        if retries > max_retries:
            raw_responses.append({
                "index": i,
                "error": "invalid_code_after_retries"
            })

    return codes, raw_responses
```

### 4.4 静态候选回退机制

```python
# 文件: src/lesr/revision_candidates.py

def generate_candidate_codes(schema: StateSchema) -> List[Tuple[str, str]]:
    """
    生成静态候选代码（LLM 采样失败时使用）
    """
    g, per_asset, field_offset = _idx_map(schema)

    # 索引计算
    close_idxs = [g + i * per_asset + field_offset["close"] for i in range(len(schema.assets))]
    open_idxs = [g + i * per_asset + field_offset["open"] for i in range(len(schema.assets))]
    high_idxs = [g + i * per_asset + field_offset["high"] for i in range(len(schema.assets))]
    low_idxs = [g + i * per_asset + field_offset["low"] for i in range(len(schema.assets))]
    holding_idxs = [g + i * per_asset + field_offset["holding"] for i in range(len(schema.assets))]

    # Candidate 0: Identity + Zero intrinsic reward
    code0 = """
import numpy as np

def revise_state(s):
    return np.array(s, dtype=float)

def intrinsic_reward(updated_s):
    return 0.0
"""

    # Candidate 1: Momentum + Exposure
    code1 = f"""
import numpy as np

CLOSE_IDXS = {close_idxs}
OPEN_IDXS = {open_idxs}
HOLDING_IDXS = {holding_idxs}

def revise_state(s):
    s = np.array(s, dtype=float)
    momentum = (s[CLOSE_IDXS] - s[OPEN_IDXS]) / (s[OPEN_IDXS] + 1e-8)
    exposure = np.sum(np.abs(s[HOLDING_IDXS]))
    updated_s = np.concatenate([s, momentum, [exposure]])
    return updated_s

def intrinsic_reward(updated_s):
    extra_start = updated_s.shape[0] - ({len(close_idxs)} + 1)
    momentum = updated_s[extra_start: extra_start + {len(close_idxs)}]
    exposure = updated_s[-1]
    return float(np.mean(momentum) - 0.01 * exposure)
"""

    # Candidate 2: Volatility + Concentration
    code2 = f"""
import numpy as np

HIGH_IDXS = {high_idxs}
LOW_IDXS = {low_idxs}
CLOSE_IDXS = {close_idxs}
HOLDING_IDXS = {holding_idxs}

def revise_state(s):
    s = np.array(s, dtype=float)
    spread = (s[HIGH_IDXS] - s[LOW_IDXS]) / (s[CLOSE_IDXS] + 1e-8)
    concentration = np.max(np.abs(s[HOLDING_IDXS]))
    updated_s = np.concatenate([s, spread, [concentration]])
    return updated_s

def intrinsic_reward(updated_s):
    extra_start = updated_s.shape[0] - ({len(close_idxs)} + 1)
    spread = updated_s[extra_start: extra_start + {len(close_idxs)}]
    concentration = updated_s[-1]
    return float(-np.mean(spread) - 0.01 * concentration)
"""

    return [
        ("identity", code0),
        ("momentum_exposure", code1),
        ("volatility_concentration", code2),
    ]
```

### 4.5 DeepSeek API 客户端

```python
# 文件: src/llm/deepseek_client.py

class DeepSeekClient:
    """
    DeepSeek API 客户端
    """
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_s: int = 60,
        use_env_proxy: bool = False
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = int(max(1, timeout_s))
        self.use_env_proxy = bool(use_env_proxy)

        # 避免 proxy 的长连接问题
        self._proxy_handler = request.ProxyHandler() \
            if self.use_env_proxy else request.ProxyHandler({})

    def _build_opener(self):
        """
        构建 opener (避免长连接)
        """
        return request.build_opener(self._proxy_handler)

    def chat(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        调用 DeepSeek Chat API
        """
        # 1. 构建 URL
        url = self.base_url
        if not url.endswith("/v1"):
            url = url + "/v1"
        url = url + "/chat/completions"

        # 2. 构建 payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # 3. 发送请求
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "llm-rl-trading/lesr",
                "Connection": "close",  # 避免长连接
            },
            method="POST",
        )

        # 4. 获取响应
        with self._build_opener().open(req, timeout=self.timeout_s) as resp:
            content = resp.read().decode("utf-8")

        # 5. 解析并返回
        obj = json.loads(content)
        return obj["choices"][0]["message"]["content"]


def from_env(
    base_url: str,
    timeout_s: int = 60,
    use_env_proxy: bool | None = None
) -> DeepSeekClient:
    """
    从环境变量创建客户端
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if use_env_proxy is None:
        use_env_proxy = _coerce_bool(
            os.environ.get("DEEPSEEK_USE_ENV_PROXY"),
            default=False
        )

    return DeepSeekClient(
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        use_env_proxy=bool(use_env_proxy),
    )
```

---

## 5. 迭代优化工作流

### 5.1 整体迭代流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LESR 迭代优化工作流 (10 轮)                              │
└─────────────────────────────────────────────────────────────────────────────┘

Iteration 0 (初始化)
 │
 ├─→ 1. 构建初始 Prompt
 │     • build_initial_prompt(task, state_desc, contract_note)
 │
 ├─→ 2. LLM 采样 k=3 个候选
 │     • sample_candidates() -> codes
 │
 ├─→ 3. 验证代码
 │     • is_valid_code() -> valid_codes
 │
 ├─→ 4. 并行训练 (4 Workers × 4 算法)
 │     • G0: Baseline
 │     • G1: Revise Only
 │     • G2: Intrinsic Only
 │     • G3: Joint
 │
 ├─→ 5. 评估性能
 │     • Sharpe Ratio, Cumulative Returns
 │
 ├─→ 6. 计算 Lipschitz 常数
 │     • 每个维度与奖励的连续性
 │
 ├─→ 7. 生成 CoT 反馈
 │     • build_cot_prompt(codes, scores, factors)
 │
 └─→ 保存历史结果

Iteration 1-9 (迭代优化)
 │
 ├─→ 1. 构建迭代 Prompt
 │     • build_next_iteration_prompt(
 │         task, state_desc,
 │         history_results,      # 上轮结果摘要
 │         history_suggestions   # 上轮建议
 │       )
 │
 ├─→ 2. 场景族采样 (Scenario Family)
 │     • trend_follow: 趋势跟踪候选
 │     • mean_revert: 均值回归候选
 │     • risk_shield: 风险屏蔽候选
 │
 ├─→ 3. LLM 采样 k=3 个候选
 │     • sample_candidates() -> codes
 │
 ├─→ 4. 验证代码
 │     • is_valid_code() -> valid_codes
 │
 ├─→ 5. 并行训练 (4 Workers × 4 算法)
 │     • 每个算法独立分支迭代
 │
 ├─→ 6. 评估性能
 │     • Sharpe Ratio, Cumulative Returns
 │
 ├─→ 7. 计算 Lipschitz 常数
 │
 ├─→ 8. 生成 CoT 反馈
 │     • build_cot_prompt(codes, scores, factors)
 │
 ├─→ 9. 更新历史
 │     • 压缩到最多 3 轮
 │     • 字符预算控制 (5000 字符)
 │
 └─→ 下一轮迭代...

Iteration 10 (最终评估)
 │
 ├─→ 1. 选择最佳候选
 │     • 每个算法独立的最佳候选
 │
 ├─→ 2. 多种子评估
 │     • seeds: [1, 2, 3, 4, 5]
 │
 ├─→ 3. 生成最终报告
 │     • 性能统计
 │     • 行为分析
 │     • 风险指标
 │
 └─→ End
```

### 5.2 场景族 (Scenario Family) 采样

```python
# 配置文件: configs/current_baseline/*.yaml

llm:
  generation_target: scenario_family  # 使用场景族采样

  scenario_family:
    enabled: true
    families:
      - trend_follow     # 趋势跟踪
      - mean_revert      # 均值回归
      - risk_shield      # 风险屏蔽
    candidates_per_family_per_iter: 1  # 每族每轮 1 个候选

┌─────────────────────────────────────────────────────────────────────────────┐
│                        场景族详细定义                                        │
└─────────────────────────────────────────────────────────────────────────────┘

1. trend_follow (趋势跟踪)
   • 目标: 捕捉价格上涨趋势
   • 特征: 动量、趋势强度、突破信号
   • intrinsic_reward: 奖励正向动量，惩罚反向持仓
   • 适用于: 牛市、强趋势市场

2. mean_revert (均值回归)
   • 目标: 捕捉价格回归均值
   • 特征: 偏离度、Z-score、布林带位置
   • intrinsic_reward: 奖励过度偏离的反向操作
   • 适用于: 震荡市场、区间交易

3. risk_shield (风险屏蔽)
   • 目标: 控制下行风险
   • 特征: 波动率、回撤、集中度、VaR
   • intrinsic_reward: 惩罚高风险敞口
   • 适用于: 熊市、高波动期
```

### 5.3 算法分支独立迭代

```python
# 配置: iteration_mode: per_algorithm_branches

llm:
  iterations: 10
  iteration_mode: per_algorithm_branches  # 每个算法独立分支
  branch_parallel_workers: 4              # 并行 worker 数

eval_algorithms:
  - a2c
  - ppo
  - sac
  - td3

┌─────────────────────────────────────────────────────────────────────────────┐
│                    算法分支独立迭代架构                                      │
└─────────────────────────────────────────────────────────────────────────────┘

                  Main Process
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
    A2C Branch    PPO Branch    SAC Branch    TD3 Branch
        │             │             │             │
        │             │             │             │
    ┌───┴───┐     ┌───┴───┐     ┌───┴───┐     ┌───┴───┐
    │ W0-9  │     │ W0-9  │     │ W0-9  │     │ W0-9  │
    └───┬───┘     └───┬───┘     └───┬───┘     └───┬───┘
        │             │             │             │
        ▼             ▼             ▼             ▼
   A2C History   PPO History   SAC History   TD3 History
   (独立)        (独立)        (独立)        (独立)

特点:
• 每个算法维护独立的候选历史
• 每个算法有自己的最佳候选
• 避免不同算法间的候选混淆
• 允许算法特定的优化策略
```

### 5.4 Lipschitz 常数计算

```python
# 概念说明 (非实际代码)

def calculate_lipschitz_constants(
    states: np.ndarray,      # (T, state_dim) 状态序列
    rewards: np.ndarray,     # (T,) 奖励序列
) -> np.ndarray:
    """
    计算每个状态维度与奖励的 Lipschitz 常数

    Lipschitz 常数定义:
    L_i = max |r_t - r_s| / |s_t[i] - s_s[i]|

    含义:
    • L_i 越小: 状态维度 i 与奖励的映射越平滑
    • L_i 越大: 状态维度 i 的微小变化会导致奖励剧烈波动
    • 最优: 关键特征维度应有较小的 L_i
    """
    state_dim = states.shape[1]
    lipschitz = np.zeros(state_dim)

    for i in range(state_dim):
        # 1. 按状态维度 i 排序
        sorted_indices = np.argsort(states[:, i])

        # 2. 计算相邻样本的奖励变化率
        delta_r = np.diff(rewards[sorted_indices])
        delta_s = np.diff(states[sorted_indices, i])

        # 3. 避免除零
        ratio = np.abs(delta_r) / (np.abs(delta_s) + 1e-6)

        # 4. 取最大值作为 Lipschitz 常数
        lipschitz[i] = np.max(ratio)

    return lipschitz


# 在训练中的应用
for episode in range(num_episodes):
    # 收集状态和奖励
    states, rewards = collect_trajectory()

    # Episode 结束后计算 Lipschitz 常数
    lipschitz = calculate_lipschitz_constants(states, rewards)

    # 保存用于 LLM 反馈
    save_lipschitz(episode, lipschitz)
```

### 5.5 候选评分与选择

```python
# 配置文件

llm:
  candidate_scoring:
    performance_mode: delta_to_g0      # 相对 G0 的增量
    performance_metric: score          # 性能指标
    performance_weight: 1.0            # 性能权重
    lipschitz_weight: 0.2              # Lipschitz 权重
    lipschitz_quantile: 0.9            # Lipschitz 分位数
    lipschitz_max_pairs: 256           # 最多对比对数
    selection_seed_count: 1            # 选择种子数

# 评分公式
score = (
    performance_weight * normalized_performance +
    lipschitz_weight * normalized_lipschitz_smoothness
)

其中:
• normalized_performance = (candidate_sharpe - g0_sharpe) / std
• normalized_lipschitz_smoothness = -quantile(lipschitz, q=0.9)

选择:
• 每个算法分支选择 score 最高的候选
• 下一轮迭代基于最佳候选生成改进版本
```

---

## 6. 与原始 LESR 的对比

### 6.1 核心差异对比表

| 维度 | 原始 LESR (机器人控制) | llm_rl_trading_finsaber (金融交易) |
|------|----------------------|----------------------------------|
| **任务类型** | 连续控制 (MuJoCo, Robotics) | 金融交易决策 |
| **环境** | Gym/MuJoCo 物理模拟器 | FINSABER 交易环境 |
| **状态空间** | 关节角度、速度、力矩 | 价格、持仓、技术指标 |
| **动作空间** | 连续 (关节力矩) | 连续 (资产权重) |
| **奖励函数** | 前进速度、能量效率 | 投资收益、风险调整 |
| **状态契约** | Generic OHLCV Schema | Native FINSABER Contract |
| **LLM 模型** | OpenAI GPT-4 | DeepSeek |
| **采样策略** | k=6, temperature=0.0 | k=3, temperature=0.2 |
| **迭代轮数** | 5 轮 | 10 轮 |
| **评估模式** | 单一模式 (G3) | 四种模式 (G0/G1/G2/G3) |
| **算法支持** | TD3 | A2C/PPO/SAC/TD3 |
| **场景族** | 无 | trend_follow/mean_revert/risk_shield |
| **先验知识** | 物理定律 | 金融理论、技术分析 |
| **Prompt 优化** | 通用模板 | trading_lesr_prior_v1 |

### 6.2 Prompt 设计对比

#### 6.2.1 System Prompt 对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    System Prompt 对比                                      │
└─────────────────────────────────────────────────────────────────────────────┘

原始 LESR (机器人控制):
─────────────────────
You are an expert in reinforcement learning and control theory.

Primary objective:
- Produce code that improves sample efficiency and final performance.

Output contract:
- Return ONLY executable Python code.
- Must contain: revise_state(s), intrinsic_reward(updated_s)
- No markdown, no explanations.

Hard constraints:
- Use only source state dimensions.
- Add 3-10 computed dimensions.
- intrinsic_reward MUST use at least one added dimension.
- Ensure numerical stability (no NaN/Inf).
- Clip intrinsic_reward to [-100, 100].

llm_rl_trading_finsaber (金融交易):
─────────────────────────────────
You are designing revise_state(s) and intrinsic_reward(updated_s)
for LESR-style trading RL.

Primary objective:
- Produce code that improves sample efficiency and risk-adjusted
  performance while preserving policy-action sensitivity.

Output contract:
- Return ONLY executable Python code.
- Must contain: revise_state(s), intrinsic_reward(updated_s)
- No markdown, no explanations, no comments.

Hard constraints:
- No future information leakage.
- NumPy-only operations.
- Keep all outputs finite and bounded.
- Keep intrinsic reward informative; avoid trivial tiny clipping.
- Intrinsic reward must stay in [-100, 100].
- **intrinsic_reward must remain meaningful when only the original
  source dims are available (G2 mode)**.
- If appended revised dims are available, intrinsic_reward should
  use them as extra context rather than as a hard dependency.
- Do not use hard-coded out-of-range indices.

Failure patterns to avoid:
- **Action-insensitive intrinsic design** dominated by state-only bias.
- **Large monotonic concentration push** that drives near-bound
  action saturation.
- **Overly sparse trigger logic** that makes reward mostly zero.
- **Unstable sign-flip terms** without confidence gating.

Empirical priors from recent experiments:
[包含大量金融交易特定的成功模式和失败案例]

Feature-group semantics:
- portfolio_memory: cash ratio, exposure, concentration, entropy
- regime: volatility, drawdown, market stress, trend strength
- dispersion: spread, rank, breadth, cross-asset disagreement
- running_risk_state: return EMA, return-squared EMA, drawdown EMA
```

#### 6.2.2 User Prompt 对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    User Prompt 对比                                        │
└─────────────────────────────────────────────────────────────────────────────┘

原始 LESR:
──────────
Task: HalfCheetah-v4 - A 2D robot cheetah running task
Objective: Maximize forward velocity while maintaining energy efficiency

State Space: 27 dimensional array s
- s[0:8]: Joint positions (8 joints)
- s[8:16]: Joint velocities
- s[16:24]: Joint torques/forces
- s[24]: Contact information (ground)
- s[25:26]: Auxiliary information

Physical Interpretation:
- The robot has 8 actuated joints
- Forward motion requires coordinated joint movements
- Energy efficiency is important
- Stability matters (avoid falling)

Your Goal: Design two functions:
1. revise_state(state) - Transform to capture:
   - Forward velocity trends
   - Energy consumption patterns
   - Coordination between joints
   - Stability indicators

2. intrinsic_reward(extended_state) - Reward:
   - Forward progress
   - Energy efficiency
   - Smooth movements

llm_rl_trading_finsaber:
────────────────────────
Task: Multi-asset portfolio trading with risk management
Objective: Maximize risk-adjusted returns while controlling downside risk

State Space: 26 dimensional array s (Native FINSABER contract)
- s[0]: Cash
- s[1:6]: Close prices (5 assets)
- s[6:11]: Holdings (5 assets)
- s[11:16]: SMA indicators (5 assets)
- s[16:21]: RSI indicators (5 assets)
- s[21:26]: Volatility indicators (5 assets)

⚠️  IMPORTANT: Native State Contract Notice
- This is NOT a generic OHLCV schema
- open/high/low/volume are NOT in the state
- indicators are in indicator-major order, not asset-major
- Close prices and holdings are separate blocks

Financial Interpretation:
- Cash is the risk-free asset
- Holdings represent current positions
- Indicators capture market regime and momentum
- Portfolio decisions affect both return and risk

Your Goal: Design two functions:
1. revise_state(state) - Transform to capture:
   - Portfolio memory (cash ratio, exposure, concentration)
   - Market regime (volatility, trend strength, drawdown)
   - Dispersion (cross-asset momentum spread)
   - Running risk state (EMA returns, turnover)

2. intrinsic_reward(state) - Reward:
   - **CRITICAL**: Must work on BOTH raw state (G2) and revised state (G3)
   - Primary signal from raw/source dims
   - Revised dims only refine, gate, or denoise
   - Avoid near-constant or almost-zero designs on raw states
   - Risk-adjusted behavior (Sharpe-ratio-like)
   - Avoid unstable portfolio-weight jumps

Constraints:
- **intrinsic_reward must have a raw-state fallback path**
- **The primary intrinsic signal must come from raw/source dims**
- **If revised extra dims are present, use them to refine the same
  signal instead of carrying the sole predictive content**
- Prefer action-relevant spread/rank/trend-confidence interactions
- Avoid state-only bias terms that don't affect action behavior
- Use confidence-gated penalties over unconditional suppression
```

### 6.3 评估机制对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    评估机制对比                                            │
└─────────────────────────────────────────────────────────────────────────────┘

原始 LESR:
──────────
评估模式:
• 单一模式: G3 (revise_state + intrinsic_reward)
• 评估指标: Cumulative Reward
• 对比基准: Baseline TD3

llm_rl_trading_finsaber:
────────────────────────
评估模式:
• G0: Baseline (无 LESR)
  - 仅环境奖励
  - 原始状态

• G1: Revise Only
  - 仅环境奖励
  - 使用 revise_state(s)

• G2: Intrinsic Only (关键差异)
  - 环境奖励 + 内在奖励
  - 原始状态 s
  - intrinsic_reward(s) ← 必须在原始状态上有效

• G3: Joint
  - 环境奖励 + 内在奖励
  - 使用 revise_state(s)
  - intrinsic_reward(revise_state(s))

评估指标:
• Sharpe Ratio (风险调整收益)
• Cumulative Returns
• Maximum Drawdown
• Action Saturation (动作饱和度)
• Intrinsic Signal Nontrivial (内在信号非平凡性)

对比分析:
──────────
• state_probe_delta_sharpe: G1 vs G0 的 Sharpe 增量
• intrinsic_probe_delta_sharpe: G2 vs G0 的 Sharpe 增量
• performance_delta_sharpe: G3 vs G0 的 Sharpe 增量

归因分析:
• 如果 state_probe_delta_sharpe > 0 且 intrinsic_probe_delta_sharpe ≈ 0:
  → 提升主要来自状态增强，内在奖励未独立生效

• 如果 state_probe_delta_sharpe ≈ 0 且 intrinsic_probe_delta_sharpe > 0:
  → 提升主要来自内在奖励，状态增强未起作用

• 如果两者都 > 0:
  → 状态增强和内在奖励都独立生效，协同效应
```

### 6.4 迭代策略对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    迭代策略对比                                            │
└─────────────────────────────────────────────────────────────────────────────┘

原始 LESR:
──────────
迭代轮数: 5 轮
采样数量: k=6 候选/轮
温度设置: temperature=0.0 (确定性)
并行策略: Tmux 并发训练
历史管理: 全部历史保留

llm_rl_trading_finsaber:
────────────────────────
迭代轮数: 10 轮
采样数量: k=3 候选/轮
温度设置: temperature=0.2 (适度探索)
并行策略: 4 Workers (进程池)
历史管理: 压缩到最近 3 轮，字符预算 5000

场景族采样:
──────────
原始 LESR: 无
llm_rl_trading_finsaber:
  • trend_follow: 趋势跟踪策略
  • mean_revert: 均值回归策略
  • risk_shield: 风险屏蔽策略
  • 每族每轮 1 个候选

算法分支:
──────────
原始 LESR: 单算法 (TD3)
llm_rl_trading_finsaber:
  • per_algorithm_branches 模式
  • A2C, PPO, SAC, TD3 独立迭代
  • 每个算法维护独立候选历史
```

---

## 7. 当前不足分析

### 7.1 架构层面的不足

#### 7.1.1 State Contract 不一致问题

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              严重问题：State Contract 不一致                                │
└─────────────────────────────────────────────────────────────────────────────┘

问题描述:
──────────
虽然 prompt 最终落盘文本已明确写入 native contract note，但迭代 CoT 的
source_dim 仍使用 schema.dim()，导致后续分析错位。

影响范围:
──────────
• build_cot_prompt() 的 source_dim 参数
• Lipschitz 常数分析的维度边界
• 额外维度统计的基准
• 候选代码的索引验证

证据:
──────────
文件: src/pipeline/demo.py
```python
# Native 分支已正确获取 raw_state_dim
raw_state_dim = native_contract.state_dim

# 但 CoT 反馈仍使用 generic schema
build_cot_prompt(
    ...,
    source_dim=schema.dim(),  # ❌ 错误！应该是 raw_state_dim
    ...
)
```

后果:
──────────
• LLM 收到的 Lipschitz 分析可能指向错误的维度
• 反馈建议基于错误的维度索引
• 迭代优化可能误导 LLM
• 验证失败率增加 (llm_errors.json 中的索引错误)
```

#### 7.1.2 System Prompt 缺少 Native 绑定

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              高优先级问题：System Prompt 通用化                              │
└─────────────────────────────────────────────────────────────────────────────┘

问题描述:
──────────
System prompt 在代码级仍是后端无感知的通用模板，native 对齐主要依赖
配置文件的 system_prompt_extra，容易在换配置时回退。

当前实现:
──────────
文件: src/lesr/prompt_templates.py
```python
def build_system_prompt(llm_cfg: dict | None = None) -> str:
    mode = str(cfg.get("system_prompt_mode", "trading_lesr_prior_v1"))

    if mode != "trading_lesr_prior_v1":
        # ❌ 回退到通用模板
        base = "You are an expert in reinforcement-learning..."
        return base

    # ✅ 使用金融交易专用模板
    base = """
    You are designing revise_state(s) and intrinsic_reward(updated_s)
    for LESR-style trading RL.
    ...
    """
    return base
```

问题:
──────────
• trading_lesr_prior_v1 模式仍是通用叙述
• Native state contract 没有硬编码到 system prompt
• 依赖外部配置传递 system_prompt_extra
• 容易被新配置覆盖或忽略

建议改进:
──────────
```python
def build_system_prompt(
    llm_cfg: dict | None = None,
    backend: str = "generic",  # ← 新增参数
    state_contract_note: str = ""  # ← 新增参数
) -> str:
    mode = str(cfg.get("system_prompt_mode", "trading_lesr_prior_v1"))

    # ✅ 根据 backend 选择模板
    if backend == "finsaber_native":
        base = _build_native_system_prompt()
    else:
        base = _build_generic_system_prompt()

    # ✅ 硬编码 state contract note
    if state_contract_note:
        base += f"\n\nAuthoritative State Contract:\n{state_contract_note}\n"

    return base
```
```

#### 7.1.3 候选验证的 Generic Schema 依赖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              中优先级问题：验证逻辑的 Schema 依赖                            │
└─────────────────────────────────────────────────────────────────────────────┘

问题描述:
──────────
revised dim delta 等诊断统计仍以 generic schema 为基线，导致搜索日志
和候选解释被污染。

影响:
──────────
• 难以判断 candidate 到底新增了哪些 native-safe 维度
• 日志显示的维度索引可能与实际不符
• 候选代码的解释性降低

当前实现:
──────────
文件: src/pipeline/demo.py
```python
# ❌ 仍使用 schema 计算修订维度
revised_dim = len(revised_state) - schema.dim()

# ✅ 应该使用 native contract
revised_dim = len(revised_state) - native_contract.state_dim
```

建议改进:
──────────
```python
# 创建统一的维度计算接口
class StateContract(ABC):
    @abstractmethod
    def state_dim(self) -> int:
        """返回原始状态维度"""
        pass

class NativeFinsaberContract(StateContract):
    def state_dim(self) -> int:
        return 1 + 2*self.stock_dim + len(self.indicators)*self.stock_dim

# 在验证逻辑中使用
revised_dim = len(revised_state) - state_contract.state_dim()
```
```

### 7.2 Intrinsic Reward 设计不足

#### 7.2.1 G2 Mode 的 Raw-State Fallback 问题

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              严重问题：Intrinsic Reward 的 Raw-State Fallback 不足          │
└─────────────────────────────────────────────────────────────────────────────┘

审计发现:
──────────
TD3 最佳候选的 intrinsic 独立贡献接近零：
• intrinsic_probe_delta_sharpe = 0.0
• intrinsic_signal_nontrivial_raw = False
• G2 行为统计几乎复制 G0
• G3 行为统计几乎复制 G1

这表明：
• intrinsic_reward(s) 在原始状态上几乎没有信号
• 所有提升都来自 revise_state
• intrinsic_reward(revise_state(s)) 只是简单地使用了 revise_state
  创建的维度，而不是在原始状态上就有意义

根本原因:
──────────
1. Prompt 要求不够明确
   • 虽然提到了 raw-state fallback，但没有强制执行
   • 没有明确的测试用例

2. LLM 生成模式
   • LLM 倾向于让 intrinsic_reward 完全依赖扩展维度
   • 原始状态的 intrinsic 分支退化为常数或接近零

3. 验证不足
   • 只验证了代码能否运行，没有验证 raw-state 分支的有效性
   • 没有检查 intrinsic_reward(raw_state) 的方差

示例问题代码:
──────────
```python
# ❌ 问题: intrinsic_reward 在原始状态上几乎为常数
def revise_state(s):
    momentum = (s[close_idxs] - s[open_idxs]) / (s[open_idxs] + 1e-8)
    return np.concatenate([s, momentum])

def intrinsic_reward(updated_s):
    # 只使用扩展维度
    momentum = updated_s[-len(close_idxs):]
    return float(np.mean(momentum) * 10.0)

# 当调用 intrinsic_reward(raw_state) 时:
# • raw_state 没有扩展维度
# • updated_s[-len(close_idxs):] 会访问错误的索引
# • 即使不报错，返回值也没有意义
```

建议改进:
──────────
1. Prompt 强化
   ```
   CRITICAL REQUIREMENT:
   Your intrinsic_reward function MUST satisfy BOTH:

   1. When called with raw state s (G2 mode):
      intrinsic_reward(s) should return a non-trivial signal
      that guides exploration WITHOUT using any revised dims.

   2. When called with revised state (G3 mode):
      intrinsic_reward(revise_state(s)) should refine the signal
      using extra dims as context, NOT as the sole source.

   Test your design:
   • What does intrinsic_reward(raw_state) return?
   • Is it non-constant? (std > 0.01)
   • Does it correlate with portfolio decisions?
   • Does it improve risk-adjusted behavior?
   ```

2. 代码验证增强
   ```python
   def validate_intrinsic_reward(code, raw_state_sample):
       """验证 intrinsic_reward 的 raw-state fallback"""
       # 1. 加载代码
       module = load_code(code)

       # 2. 测试 raw-state 分支
       raw_reward = module.intrinsic_reward(raw_state_sample)

       # 3. 检查非平凡性
       raw_rewards = []
       for _ in range(100):
           s = np.random.randn(*raw_state_sample.shape)
           raw_rewards.append(module.intrinsic_reward(s))

       raw_rewards = np.array(raw_rewards)

       # 4. 验证方差
       if np.std(raw_rewards) < 0.01:
           return False, "raw-state intrinsic reward is nearly constant"

       # 5. 验证范围
       if np.any(np.abs(raw_rewards) > 100):
           return False, "raw-state intrinsic reward exceeds [-100, 100]"

       return True, "valid"
   ```

3. 评估分离
   ```python
   # 明确分离 G2 和 G3 的评估
   metrics = {
       "g2_sharpe": evaluate(revise_state=None, intrinsic_reward=intrinsic_reward),
       "g3_sharpe": evaluate(revise_state=revise_state, intrinsic_reward=intrinsic_reward),
       "intrinsic_independence": g2_sharpe - g0_sharpe,
   }

   # 只有当 intrinsic_independence > threshold 时才认为有效
   if metrics["intrinsic_independence"] < 0.01:
       logger.warning("intrinsic reward does not work independently")
   ```
```

#### 7.2.2 Action-Insensitive 设计问题

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              高优先级问题：Action-Insensitive Intrinsic Reward              │
└─────────────────────────────────────────────────────────────────────────────┘

问题描述:
──────────
审计发现大量 intrinsic_reward 设计存在 action-insensitive 问题：
• 只依赖状态级别信息（如市场趋势）
• 不考虑当前持仓或动作
• 导致内在奖励不能引导策略改进

示例问题代码:
──────────
```python
# ❌ 问题: intrinsic_reward 只依赖市场状态，与持仓无关
def intrinsic_reward(updated_s):
    market_trend = np.mean(updated_s[close_idxs])  # 市场平均价格
    return float(market_trend)  # 只奖励市场上涨，不管持仓

# 问题:
# • 做多时，市场上涨 → 正奖励 ✅
# • 做空时，市场上涨 → 正奖励 ❌ (应该是负奖励)
# • 空仓时，市场上涨 → 正奖励 ❌ (没有持仓，不应该奖励)
```

正确设计:
──────────
```python
# ✅ 正确: 考虑持仓和市场状态的交互
def intrinsic_reward(updated_s):
    # 市场信号
    market_trend = np.mean(updated_s[close_idxs])

    # 当前持仓
    holdings = updated_s[holding_idxs]
    exposure = np.sum(holdings)

    # 交互项：持仓 × 市场趋势
    direction_correctness = np.sign(exposure) * market_trend

    # 奖励方向正确的持仓
    return float(direction_correctness * 10.0)
```

System Prompt 已有警告，但仍需加强:
──────────
```
Failure patterns to AVOID:
- Action-insensitive intrinsic design dominated by state-only bias terms.
  Example: reward only based on market trend without considering positions.

Your intrinsic_reward should:
- Change when portfolio composition changes
- Reward correct positioning (long in up-trend, short in down-trend)
- Penalize misaligned positions
- Be sensitive to action decisions
```
```

### 7.3 迭代优化策略不足

#### 7.3.1 历史压缩导致的信息丢失

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              中优先级问题：历史压缩的信息丢失                                │
└─────────────────────────────────────────────────────────────────────────────┘

当前实现:
──────────
• _PROMPT_HISTORY_MAX_ITERS = 3 (只保留最近 3 轮)
• _PROMPT_HISTORY_TOTAL_CHAR_BUDGET = 5000 (总预算 5000 字符)
• _PROMPT_HISTORY_MAX_RESULT_CHARS = 1200 (结果摘要 1200 字符)
• _PROMPT_HISTORY_MAX_SUGGESTION_CHARS = 700 (建议摘要 700 字符)

问题:
──────────
1. 早期成功模式可能被丢弃
   • 迭代 1 的优秀候选可能在迭代 5 被遗忘
   • 无法回溯到早期成功模式

2. 压缩算法可能丢失关键信息
   • _compact_history_text 只保留前 N 行
   • 可能截断关键的代码段

3. 字符预算可能不足
   • 复杂的候选代码可能超过 1200 字符
   • 压缩后可能丢失函数细节

示例:
──────────
```python
# 原始结果 (1500 字符)
Former Iteration:3 Summary
Performance: Sharpe=1.23, Cumulative Return=45.6%
Code:
import numpy as np

def revise_state(s):
    # [复杂的特征工程，100 行代码]
    ...

# 压缩后 (800 字符)
Former Iteration:3 Summary
Performance: Sharpe=1.23, Cumulative Return=45.6%
Code:
import numpy as np
def revise_state(s):
    # [只保留前 10 行，丢失关键逻辑]
    momentum = ...
```

建议改进:
──────────
1. 智能压缩
   ```python
   def _smart_compress_code(code: str, max_chars: int) -> str:
       """
       智能压缩代码，保留关键部分
       """
       # 1. 提取函数签名
       functions = extract_function_signatures(code)

       # 2. 提取关键特征计算
       key_features = extract_key_features(code)

       # 3. 提取奖励设计
       reward_design = extract_reward_design(code)

       # 4. 组合
       compressed = f"""
       Functions: {functions}
       Key Features: {key_features}
       Reward Design: {reward_design}
       """

       # 5. 裁剪到预算
       return _trim_block(compressed, max_chars)
   ```

2. 分层历史
   ```python
   # 保留完整历史（内部）
   full_history = [
       {"iteration": 0, "code": ..., "performance": ...},
       {"iteration": 1, "code": ..., "performance": ...},
       ...
   ]

   # 生成 Prompt 时使用压缩历史
   compressed_history = _compress_for_prompt(full_history)

   # 但 CoT 分析时使用完整历史
   cot_feedback = _generate_cot_with_full_history(full_history)
   ```

3. 关键模式持久化
   ```python
   # 提取并保留成功模式
   successful_patterns = extract_successful_patterns(full_history)

   # 在 Prompt 中明确列出
   prompt += f"""
   Successful patterns identified across all iterations:
   {format_successful_patterns(successful_patterns)}
   """
   ```
```

#### 7.3.2 场景族采样的局限性

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              中优先级问题：场景族采样的局限性                                │
└─────────────────────────────────────────────────────────────────────────────┘

当前实现:
──────────
scenario_family:
  families:
    - trend_follow
    - mean_revert
    - risk_shield
  candidates_per_family_per_iter: 1

问题:
──────────
1. 场景族定义过于粗糙
   • 只有 3 个大类，无法覆盖复杂市场环境
   • 没有考虑市场状态转换（牛市→熊市）
   • 没有考虑跨资产相关性

2. 每族每轮只有 1 个候选
   • 探索不足
   • 可能错过该族的更好设计

3. 场景族与 Prompt 的连接不明确
   • Prompt 没有明确说明当前是哪个族
   • LLM 可能生成不符合场景族的代码

示例:
──────────
```python
# 当前实现 (implicit)
# 场景族信息在配置中，但没有传递给 Prompt

# 建议改进 (explicit)
def build_scenario_specific_prompt(
    scenario_family: str,
    task_description: str,
    state_desc: List[str],
) -> str:
    """
    构建场景族特定的 Prompt
    """
    if scenario_family == "trend_follow":
        extra_instruction = """
        This candidate should focus on trend-following strategies:
        - Capture positive momentum
        - Reward alignment with market trend
        - Use trend strength indicators
        - Penalize counter-trend positions
        """

    elif scenario_family == "mean_revert":
        extra_instruction = """
        This candidate should focus on mean-reversion strategies:
        - Capture overbought/oversold conditions
        - Reward contrarian positions
        - Use statistical deviation measures
        - Penalize trend-chasing behavior
        """

    elif scenario_family == "risk_shield":
        extra_instruction = """
        This candidate should focus on risk management:
        - Limit downside exposure
        - Reward diversification
        - Use volatility and drawdown measures
        - Penalize concentration and tail risk
        """

    return f"""
    {base_prompt}

    Scenario Family: {scenario_family}
    {extra_instruction}
    """
```

建议改进:
──────────
1. 扩展场景族
   ```python
   scenario_family:
     families:
       - trend_follow:
           subtypes:
             - momentum_breakout
             - moving_average_crossover
             - trend_strength
       - mean_revert:
           subtypes:
             - rsi_overbought_oversold
             - bollinger_band_reversion
             - statistical_arbitrage
       - risk_shield:
           subtypes:
             - volatility_targeting
             - drawdown_control
             - tail_risk_hedging
       - regime_switching:  # 新增
           subtypes:
             - bull_bear_switch
             - volatility_regime
             - correlation_regime
   ```

2. 增加候选数量
   ```python
   candidates_per_family_per_iter: 2  # 从 1 增加到 2
   ```

3. 场景族特定验证
   ```python
   def validate_scenario_alignment(
       code: str,
       scenario_family: str
   ) -> Tuple[bool, str]:
       """
       验证代码是否符合场景族
       """
       if scenario_family == "trend_follow":
           # 检查是否有趋势相关特征
           if "momentum" not in code and "trend" not in code:
               return False, "trend_follow candidate should use momentum/trend"

       elif scenario_family == "mean_revert":
           # 检查是否有均值回归相关特征
           if "mean" not in code and "deviation" not in code:
               return False, "mean_revert candidate should use mean/deviation"

       return True, "valid"
   ```
```

### 7.4 DRL Backend 集成不足

#### 7.4.1 算法特定适配缺失

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              中优先级问题：算法特定适配缺失                                  │
└─────────────────────────────────────────────────────────────────────────────┘

审计发现:
──────────
• TD3: 最佳候选的 intrinsic 独立贡献接近零
• PPO: 部分候选能得到非零 intrinsic 增益
• SAC: 名义上是 intrinsic-first，但独立增益几乎为零
• A2C: 提升主要来自 revise/state path

问题:
──────────
不同 DRL 算法对 intrinsic reward 的敏感度不同：
• TD3: 对噪声敏感，需要平滑的 intrinsic 信号
• PPO: 对 clipped reward 敏感，intrinsic 不能过大
• SAC: 对 exploration bonus 敏感，intrinsic 应引导探索
• A2C: 对 variance 敏感，intrinsic 需要稳定

当前实现:
──────────
所有算法使用相同的 intrinsic_reward 设计，没有算法特定适配。

System Prompt 中的要求:
──────────
```
Mechanism requirements from TD3 diagnostics:
- Keep concentration/bound penalties smooth and state-dependent
- Prefer confidence-gated penalties over unconditional suppression
- Encourage action-relevant ranking terms, not only level shifts
- Avoid designs that change reward totals but leave action behavior
  nearly unchanged
```

但这些要求是 TD3 特定的，没有扩展到其他算法。

建议改进:
──────────
1. 算法特定 System Prompt
   ```python
   def build_system_prompt(
       llm_cfg: dict,
       algorithm: str,  # ← 新增参数
   ) -> str:
       base_prompt = _get_base_prompt()

       if algorithm == "td3":
           algorithm_specific = """
           TD3-specific requirements:
           - Smooth intrinsic signals (avoid discontinuities)
           - Use confidence-gated penalties
           - Avoid reward shaping that only changes total reward
             without affecting action gradients
           """

       elif algorithm == "ppo":
           algorithm_specific = """
           PPO-specific requirements:
           - Keep intrinsic reward in moderate range [-10, 10]
           - Avoid large intrinsic rewards that cause clipping
           - Use probability-weighted intrinsic signals
           """

       elif algorithm == "sac":
           algorithm_specific = """
           SAC-specific requirements:
           - Intrinsic reward should guide exploration
           - Use entropy-aware intrinsic signals
           - Reward novel states/actions
           """

       elif algorithm == "a2c":
           algorithm_specific = """
           A2C-specific requirements:
           - Keep intrinsic reward stable (low variance)
           - Use baseline-subtracted intrinsic signals
           - Avoid intrinsic rewards that amplify variance
           """

       return base_prompt + "\n\n" + algorithm_specific
   ```

2. 算法特定验证
   ```python
   def validate_intrinsic_for_algorithm(
       code: str,
       algorithm: str,
       sample_states: np.ndarray,
   ) -> Tuple[bool, str]:
       """
       验证 intrinsic_reward 是否适合特定算法
       """
       # 测试 intrinsic_reward 的统计特性
       rewards = []
       for s in sample_states:
           r = intrinsic_reward(s)
           rewards.append(r)

       rewards = np.array(rewards)

       if algorithm == "td3":
           # TD3 需要平滑的信号
           gradient = np.gradient(rewards)
           if np.std(gradient) > threshold:
               return False, "intrinsic reward too noisy for TD3"

       elif algorithm == "ppo":
           # PPO 需要适度的范围
           if np.max(np.abs(rewards)) > 10:
               return False, "intrinsic reward too large for PPO"

       elif algorithm == "sac":
           # SAC 需要探索性信号
           if np.std(rewards) < 0.1:
               return False, "intrinsic reward not exploratory enough for SAC"

       return True, "valid"
   ```
```

---

## 8. 改进建议

### 8.1 核心改进：重新设计金融 DRL 的 Prompt 和 Objective

#### 8.1.1 问题根源：从机器人到金融的思维转换不足

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              核心问题：LESR 的"机器人思维"未完全转换到"金融思维"            │
└─────────────────────────────────────────────────────────────────────────────┘

原始 LESR (机器人控制) 的思维模式:
─────────────────────────────
• 状态: 关节角度、速度、力矩
• 目标: 前进速度、能量效率
• 约束: 物理定律（运动学、动力学）
• 奖励: 连续、平滑、可微分
• 特征: 物理量（动能、势能、角动量）
• 优化: 梯度下降、策略梯度

金融 DRL 的思维模式:
───────────────────
• 状态: 价格、持仓、技术指标、市场状态
• 目标: 风险调整收益、夏普比率
• 约束: 交易成本、仓位限制、风险约束
• 奖励: 稀疏、噪声、非平稳、胖尾
• 特征: 金融量（动量、波动率、相关性、风险暴露）
• 优化: 资产配置、风险分散、动态对冲

当前项目的问题:
───────────────
虽然 prompt 已经包含金融特定内容，但底层逻辑仍是"机器人控制"：

1. ❌ 把 intrinsic_reward 当作"探索奖励"
   • 机器人: 鼓励访问新状态（探索）
   • 金融: 应该引导风险-aware 的决策

2. ❌ 把 revise_state 当作"特征增强"
   • 机器人: 添加物理特征（能量、协调）
   • 金融: 应该暴露风险暴露和投资组合约束

3. ❌ 把 Lipschitz 当作"平滑性"
   • 机器人: 奖励函数的连续性
   • 金融: 应该是风险-收益权衡的稳定性

4. ❌ G2/G3 模式的误解
   • 当前: G2 是"intrinsic only"，G3 是"joint"
   • 应该: G2 测试 raw-state 的预测能力，G3 测试 revised-state 的增强效果
```

#### 8.1.2 重新设计 Objective：从"探索奖励"到"风险感知引导"

```python
# 新的 Objective 设计

## 原始 Objective (机器人控制)
objective = "maximize cumulative reward + exploration bonus"

## 新的 Objective (金融 DRL)
objective = """
maximize risk-adjusted returns (Sharpe Ratio)
subject to:
  - position limits
  - turnover constraints
  - drawdown limits
  - transaction costs

using:
  - revise_state: expose risk exposures and portfolio constraints
  - intrinsic_reward: guide risk-aware decisions, not exploration
"""

## 核心差异

### 原始 LESR
intrinsic_reward(s) = exploration_bonus(s)
  • 鼓励访问新状态
  • 奖励多样性
  • 辅助环境奖励

### 新设计 (金融 DRL)
intrinsic_reward(s) = risk_aware_guidance(s)
  • 惩罚过度风险暴露
  • 奖励风险调整后的收益
  • 引导多样化配置
  • 考虑交易成本
  • 动态风险预算

## 具体设计

def intrinsic_reward(state):
    """
    Risk-aware intrinsic reward for portfolio management

    NOT an exploration bonus, but a risk management signal
    """
    # 1. 风险暴露惩罚
    concentration_penalty = -calculate_concentration_risk(state)
    volatility_penalty = -calculate_volatility_exposure(state)

    # 2. 风险调整收益奖励
    risk_adjusted_return = calculate_sharpe_like_signal(state)

    # 3. 多样化奖励
    diversification_bonus = calculate_diversification_benefit(state)

    # 4. 交易成本惩罚
    turnover_penalty = -calculate_turnover_cost(state)

    # 5. 综合信号
    total = (
        risk_adjusted_return +
        diversification_bonus -
        concentration_penalty -
        volatility_penalty -
        turnover_penalty
    )

    # 6. 归一化到合理范围
    return np.clip(total, -100, 100)
```

#### 8.1.3 重新设计 Prompt：从"物理特征"到"金融特征"

```python
# 新的 System Prompt 设计

def build_financial_drl_system_prompt(
    algorithm: str,
    objective: str = "risk_adjusted_returns",
) -> str:
    return f"""
You are a quantitative finance expert designing state representation and
risk-aware reward signals for portfolio management using Deep Reinforcement
Learning (DRL).

**IMPORTANT**: This is NOT robotics control. This is financial trading.

**Objective**:
{objective}

**Core Principles**:

1. Risk-First Thinking
   - All decisions should consider risk-adjusted returns, not raw returns
   - Reward volatility-adjusted performance (Sharpe Ratio, not total return)
   - Penalize excessive risk-taking, even if profitable

2. Portfolio Constraints
   - Position limits: no single asset should dominate
   - Turnover constraints: avoid excessive trading
   - Drawdown limits: control downside risk
   - Transaction costs: trading is not free

3. Market Regime Awareness
   - Bull markets: reward trend-following with risk controls
   - Bear markets: reward defensive positioning
   - High volatility: reward reduced exposure and diversification
   - Low volatility: reward strategic risk-taking

4. Diversification Benefit
   - Reward uncorrelated returns across assets
   - Penalize high concentration in few assets
   - Consider cross-asset correlations and spillovers

**State Representation (revise_state)**:

Should expose FINANCIAL features, not physical features:

✅ GOOD (Financial Features):
- portfolio_memory: cash ratio, exposure, concentration, entropy
- regime: volatility level, trend strength, drawdown, market stress
- dispersion: cross-asset momentum spread, winner-minus-loser
- risk_measures: VaR, CVaR, beta, correlation with market
- cost_awareness: unrealized P&L, transaction cost estimates

❌ BAD (Physical/Robotics Features):
- Energy, momentum (physics), velocity, acceleration
- Coordination, stability (unless financial interpretation)
- Generic mathematical transforms without financial meaning

**Intrinsic Reward (intrinsic_reward)**:

Should guide RISK-AWARE decisions, NOT exploration:

✅ GOOD (Risk-Aware Guidance):
- Positive term: risk-adjusted return (return / volatility)
- Positive term: diversification benefit (low correlation)
- Negative term: concentration penalty (high exposure to few assets)
- Negative term: volatility penalty (high portfolio volatility)
- Negative term: drawdown penalty (recent losses)
- Negative term: turnover penalty (excessive trading)

❌ BAD (Exploration Bonus):
- Reward visiting new states (this is exploration, not risk management)
- Reward diversity without considering risk
- Pure novelty detection without financial interpretation

**Algorithm-Specific Requirements**:

For {algorithm.upper()}:
{get_algorithm_specific_requirements(algorithm)}

**Output Contract**:
- Return ONLY executable Python code
- Must contain: import numpy as np, revise_state(s), intrinsic_reward(s)
- No markdown, no explanations
- Ensure numerical stability (no NaN/Inf)
- Clip intrinsic_reward to [-100, 100]

**CRITICAL: Raw-State Fallback**:
intrinsic_reward MUST work on raw state alone (G2 mode):
- Test: what does intrinsic_reward(raw_state) return?
- Should be: non-constant, risk-aware, guides decisions
- Should NOT be: near-zero, pure exploration, state-only bias
"""


def get_algorithm_specific_requirements(algorithm: str) -> str:
    """获取算法特定要求"""
    if algorithm == "td3":
        return """
- TD3 is sensitive to reward noise
- Use smooth, continuous intrinsic signals (avoid hard thresholds)
- Prefer confidence-gated penalties (e.g., max(0, x - threshold))
- Avoid intrinsic rewards that change total reward without affecting
  action gradients (action-insensitive designs)
- Ensure intrinsic_reward changes when portfolio composition changes
"""
    elif algorithm == "ppo":
        return """
- PPO uses clipped policy objective
- Keep intrinsic_reward in moderate range [-10, 10] to avoid clipping
- Use probability-weighted intrinsic signals
- Avoid large intrinsic rewards that dominate environment reward
- Consider baseline subtraction for stability
"""
    elif algorithm == "sac":
        return """
- SAC maximizes entropy for exploration
- Intrinsic reward should guide WHAT to explore, not encourage exploration
- Use entropy-aware intrinsic signals (reward diversity, not chaos)
- Reward novel but risk-aware portfolios
- Avoid intrinsic rewards that encourage random exploration
"""
    elif algorithm == "a2c":
        return """
- A2C uses advantage function
- Keep intrinsic_reward stable (low variance)
- Use baseline-subtracted intrinsic signals
- Avoid intrinsic rewards that amplify variance
- Consider running averages for stability
"""
    else:
        return ""
```

#### 8.1.4 重新设计 User Prompt：从"任务描述"到"金融场景"

```python
# 新的 User Prompt 设计

def build_financial_drl_user_prompt(
    task_description: str,
    state_desc: List[str],
    state_contract_note: str,
    market_regime: str,  # ← 新增
    objective: str,      # ← 新增
) -> str:
    return f"""
**Task**: {task_description}

**Objective**: {objective}

**Market Regime**: {market_regime}

**Current State**: {total_dim}-dimensional array s

**State Details**:
{detail_content}

**State Contract Note**:
{state_contract_note}

**Your Mission**:

Design TWO functions for RISK-AWARE portfolio management:

1. **revise_state(s)** - Expose financial risk features
   Input: Raw state s (cash, prices, holdings, indicators)
   Output: Extended state with risk-aware features

   Should compute:
   - Portfolio memory: cash ratio, exposure, concentration, entropy
   - Regime indicators: volatility, trend strength, drawdown
   - Risk measures: VaR, beta, correlation with market
   - Cost awareness: unrealized P&L, transaction cost estimates

   Should NOT compute:
   - Generic mathematical transforms (sin, cos, exp) without financial meaning
   - Physical quantities (energy, momentum in physics sense)
   - Irrelevant statistical aggregates

2. **intrinsic_reward(s)** - Guide risk-aware decisions
   Input: State (raw OR revised)
   Output: Scalar in [-100, 100]

   Should implement:
   - Positive: risk-adjusted return (return / volatility)
   - Positive: diversification benefit (low correlation)
   - Negative: concentration penalty (high exposure)
   - Negative: volatility penalty (high portfolio vol)
   - Negative: drawdown penalty (recent losses)
   - Negative: turnover penalty (excessive trading)

   Should NOT implement:
   - Exploration bonus (this is not exploration)
   - Pure novelty detection
   - State-only bias (doesn't affect actions)
   - Hard thresholds without financial interpretation

**CRITICAL: Raw-State Fallback**:

intrinsic_reward MUST work on BOTH:
1. Raw state (G2 mode): intrinsic_reward(raw_state)
2. Revised state (G3 mode): intrinsic_reward(revise_state(raw_state))

Test your design:
- What does intrinsic_reward(raw_state) return?
- Is it non-constant? (std > 0.01)
- Does it guide risk-aware decisions?
- Does it reward diversification and penalize concentration?
- Does it consider transaction costs?

**Market Regime Context**: {market_regime}

Consider:
- Bull market: reward trend-following with risk controls
- Bear market: reward defensive positioning
- High volatility: reward reduced exposure and diversification
- Low volatility: reward strategic risk-taking

**Numerical Stability**:
- All divisions must be safe (add epsilon or use fallback)
- Clip all outputs to valid ranges
- Avoid NaN/Inf at all costs
- Use bounded transforms (tanh, clip, safe division)

**Output Format**:
```python
import numpy as np

def revise_state(s):
    # Your implementation
    return extended_state

def intrinsic_reward(s):
    # Your implementation (must work on raw state too!)
    return reward_value
```

Generate clean, well-structured code.
"""
```

#### 8.1.5 示例：正确的金融 DRL 候选代码

```python
# ✅ 正确示例: 金融 DRL 的 revise_state 和 intrinsic_reward

import numpy as np

# 金融特征索引 (假设 native contract)
CASH_IDX = 0
CLOSE_IDXS = slice(1, 6)       # 5 个资产
HOLDING_IDXS = slice(6, 11)    # 5 个资产
SMA_IDXS = slice(11, 16)       # SMA 指标
RSI_IDXS = slice(16, 21)       # RSI 指标
VOL_IDXS = slice(21, 26)       # 波动率指标

def revise_state(s):
    """
    金融风险感知的状态表示增强

    目标: 暴露投资组合的风险暴露和约束，而不是"物理特征"
    """
    s = np.array(s, dtype=float)

    # ===== 基础数据 =====
    cash = s[CASH_IDX]
    close_prices = s[CLOSE_IDXS]
    holdings = s[HOLDING_IDXS]
    sma = s[SMA_IDXS]
    rsi = s[RSI_IDXS]
    vol = s[VOL_IDXS]

    # ===== Portfolio Memory (投资组合记忆) =====

    # 1. 现金比例
    total_value = cash + np.sum(close_prices * holdings)
    cash_ratio = cash / (total_value + 1e-8)

    # 2. 敞口 ( invested wealth / total wealth )
    invested_value = np.sum(close_prices * np.abs(holdings))
    exposure = invested_value / (total_value + 1e-8)

    # 3. 集中度 (最大持仓比例)
    position_values = np.abs(close_prices * holdings)
    concentration = np.max(position_values) / (total_value + 1e-8)

    # 4. 熵 (分散度)
    weights = position_values / (total_value + 1e-8)
    weights = weights[weights > 0]  # 只考虑非零持仓
    entropy = -np.sum(weights * np.log(weights + 1e-8))

    # 5. 再平衡压力 (当前配置与目标配置的距离)
    # 假设目标是等权重配置
    target_weights = np.ones(len(holdings)) / len(holdings)
    current_weights = weights
    rebalancing_pressure = np.linalg.norm(current_weights - target_weights)

    # ===== Regime (市场状态) =====

    # 6. 市场波动率 (跨资产平均波动率)
    market_volatility = np.mean(vol)

    # 7. 波动率比值 (当前 / 历史)
    volatility_ratio = market_volatility / (np.mean(vol) + 1e-8)

    # 8. 趋势强度 (价格相对于 SMA 的偏离)
    trend_strength = np.mean((close_prices - sma) / (sma + 1e-8))

    # 9. 动量信号 (过去收益的代理)
    # 假设 RSI > 50 表示上涨趋势
    momentum_signal = np.mean((rsi - 50) / 50.0)

    # 10. 市场压力 (结合波动率和趋势)
    # 高波动 + 负趋势 = 高压力
    market_stress = market_volatility * (1.0 - np.clip(trend_strength, -1, 1))

    # ===== Dispersion (离散度) =====

    # 11. 跨资产动量离散度
    asset_momentum = (close_prices - sma) / (sma + 1e-8)
    momentum_dispersion = np.std(asset_momentum)

    # 12. 赢家减输家 (最大动量 - 最小动量)
    winner_minus_loser = np.max(asset_momentum) - np.min(asset_momentum)

    # 13. 正向动量占比 (动量 > 0 的资产占比)
    positive_momentum_ratio = np.mean(asset_momentum > 0)

    # ===== Risk Measures (风险度量) =====

    # 14. 投资组合方差 (简化估计，忽略协方差)
    portfolio_variance = np.sum((weights * vol) ** 2)

    # 15. Beta (相对于市场的敏感性)
    # 假设"市场"是等权重组合
    market_return = np.mean(asset_momentum)
    cov_with_market = np.mean(weights * asset_momentum) - market_return
    var_market = np.var(asset_momentum) + 1e-8
    beta = cov_with_market / var_market

    # 16. 价值风险 (VaR) - 简化版本
    # 假设正态分布，95% VaR
    portfolio_std = np.sqrt(portfolio_variance)
    var_95 = -1.65 * portfolio_std

    # ===== Cost Awareness (成本意识) =====

    # 17. 未实现损益
    unrealized_pnl = np.sum(holdings * (close_prices - sma))

    # 18. 交易成本估计 (假设再平衡到等权重)
    target_notional = total_value / len(holdings)
    current_notional = close_prices * holdings
    trades = np.abs(current_notional - target_notional)
    transaction_cost = 0.001 * np.sum(trades)  # 0.1% 交易成本

    # ===== 组合扩展状态 =====
    extended_features = np.array([
        # Portfolio Memory (5 dims)
        cash_ratio,
        exposure,
        concentration,
        entropy,
        rebalancing_pressure,

        # Regime (5 dims)
        market_volatility,
        volatility_ratio,
        trend_strength,
        momentum_signal,
        market_stress,

        # Dispersion (3 dims)
        momentum_dispersion,
        winner_minus_loser,
        positive_momentum_ratio,

        # Risk Measures (3 dims)
        portfolio_variance,
        beta,
        var_95,

        # Cost Awareness (2 dims)
        unrealized_pnl / (total_value + 1e-8),  # 归一化
        transaction_cost / (total_value + 1e-8),  # 归一化
    ])

    return np.concatenate([s, extended_features])


def intrinsic_reward(s):
    """
    风险感知的内在奖励

    关键: 必须在原始状态 (raw state) 上也有效！

    设计原则:
    1. 风险调整收益 (不是探索奖励)
    2. 多样化奖励
    3. 集中度惩罚
    4. 波动率惩罚
    5. 交易成本惩罚
    """
    s = np.array(s, dtype=float)

    # ===== 原始状态数据 =====
    cash = s[CASH_IDX]
    close_prices = s[CLOSE_IDXS]
    holdings = s[HOLDING_IDXS]
    vol = s[VOL_IDXS]

    total_value = cash + np.sum(close_prices * holdings)
    position_values = close_prices * holdings
    weights = position_values / (total_value + 1e-8)

    # ===== 如果有扩展特征，使用它们 =====
    if len(s) > 26:
        # 提取扩展特征
        idx = 26
        cash_ratio = s[idx]; idx += 1
        exposure = s[idx]; idx += 1
        concentration = s[idx]; idx += 1
        entropy = s[idx]; idx += 1
        market_volatility = s[idx]; idx += 1
        volatility_ratio = s[idx]; idx += 1
        trend_strength = s[idx]; idx += 1
        momentum_signal = s[idx]; idx += 1
        momentum_dispersion = s[idx]; idx += 1
        portfolio_variance = s[idx]; idx += 1
        beta = s[idx]; idx += 1
    else:
        # 从原始状态计算简化版本
        cash_ratio = cash / (total_value + 1e-8)
        exposure = np.sum(np.abs(position_values)) / (total_value + 1e-8)
        concentration = np.max(np.abs(weights))
        entropy = -np.sum(weights * np.log(np.abs(weights) + 1e-8))
        market_volatility = np.mean(vol)
        volatility_ratio = market_volatility / (np.mean(vol) + 1e-8)
        trend_strength = np.mean(close_prices) / (np.mean(close_prices) + 1e-8) - 1.0
        momentum_signal = 0.0  # 无法从原始状态计算
        momentum_dispersion = np.std(close_prices) / (np.mean(close_prices) + 1e-8)
        portfolio_variance = np.sum((weights * vol) ** 2)
        beta = 1.0  # 简化假设

    # ===== 1. 风险调整收益 (正项) =====
    # 使用趋势强度作为收益代理
    # 高趋势 × 低波动 = 高风险调整收益
    risk_adjusted_return = trend_strength / (market_volatility + 1e-8)
    reward_risk_adjusted = 50.0 * np.tanh(risk_adjusted_return)

    # ===== 2. 多样化奖励 (正项) =====
    # 低集中度 + 高熵 = 高多样化
    diversification_score = (1.0 - concentration) * entropy
    reward_diversification = 20.0 * diversification_score

    # ===== 3. 集中度惩罚 (负项) =====
    # 使用二次惩罚，避免过度集中
    penalty_concentration = -30.0 * (concentration ** 2)

    # ===== 4. 波动率惩罚 (负项) =====
    # 高波动率 → 高惩罚
    penalty_volatility = -20.0 * market_volatility

    # ===== 5. 交易成本惩罚 (负项) =====
    # 使用 turnover 的代理 (假设有扩展特征)
    if len(s) > 26:
        # 如果有交易成本特征，使用它
        transaction_cost_ratio = s[idx - 2]  # 倒数第二个特征
        penalty_turnover = -10.0 * transaction_cost_ratio
    else:
        # 否则使用简化版本
        # 估计 turnover: 当前权重与等权重的距离
        target_weights = np.ones(len(weights)) / len(weights)
        turnover = np.linalg.norm(weights - target_weights)
        penalty_turnover = -10.0 * turnover

    # ===== 6. 市场状态交互 (动态调整) =====
    # 牛市: 奖励趋势跟踪
    # 熊市: 奖励防御性
    if trend_strength > 0:
        # 牛市: 奖励适度敞口
        reward_exposure = 10.0 * np.clip(exposure - 0.5, -0.5, 0.5)
    else:
        # 熊市: 奖励低敞口
        reward_exposure = -10.0 * exposure

    # ===== 综合奖励 =====
    total_reward = (
        reward_risk_adjusted +      # 风险调整收益
        reward_diversification +    # 多样化
        penalty_concentration +     # 集中度惩罚
        penalty_volatility +        # 波动率惩罚
        penalty_turnover +          # 交易成本惩罚
        reward_exposure             # 敞口奖励 (市场状态依赖)
    )

    # ===== 归一化到 [-100, 100] =====
    return np.clip(total_reward, -100, 100)
```

### 8.2 其他改进建议

#### 8.2.1 增强 State Contract 验证

```python
# 创建统一的 State Contract 接口

from abc import ABC, abstractmethod

class StateContract(ABC):
    """状态契约接口"""

    @abstractmethod
    def state_dim(self) -> int:
        """返回原始状态维度"""
        pass

    @abstractmethod
    def describe_compact(self) -> List[str]:
        """返回状态描述（用于 Prompt）"""
        pass

    @abstractmethod
    def prompt_note(self) -> str:
        """返回 Prompt 中的契约说明"""
        pass

    @abstractmethod
    def validate_indices(self, code: str) -> Tuple[bool, str]:
        """验证代码中的索引是否有效"""
        pass


class NativeFinsaberContract(StateContract):
    """FINSABER Native 状态契约"""

    def __init__(self, stock_dim: int, indicators: List[str]):
        self.stock_dim = stock_dim
        self.indicators = indicators
        self.indicator_dim = len(indicators)

    def state_dim(self) -> int:
        """
        Native state 维度公式:
        1 (cash) + 2×stock_dim (close + holdings) +
        indicator_dim×stock_dim (indicators in indicator-major order)
        """
        return 1 + 2 * self.stock_dim + self.indicator_dim * self.stock_dim

    def describe_compact(self) -> List[str]:
        """生成状态描述"""
        desc = [
            f"s[0]: cash (risk-free asset)",
        ]

        # Close prices
        for i in range(self.stock_dim):
            desc.append(f"s[{1+i}]: close price of asset {i}")

        # Holdings
        for i in range(self.stock_dim):
            desc.append(f"s[{1+self.stock_dim+i}]: holdings of asset {i}")

        # Indicators (indicator-major order)
        base = 1 + 2 * self.stock_dim
        for idx, indicator in enumerate(self.indicators):
            for i in range(self.stock_dim):
                desc.append(
                    f"s[{base + idx*self.stock_dim + i}]: "
                    f"{indicator} of asset {i}"
                )

        return desc

    def prompt_note(self) -> str:
        """生成 Prompt 说明"""
        return f"""
⚠️  AUTHORITATIVE NATIVE STATE CONTRACT ⚠️

This is NOT a generic OHLCV schema. The native FINSABER state has:

1. State dimension formula:
   total_dim = 1 + 2×stock_dim + len(indicators)×stock_dim
   current_dim = 1 + 2×{self.stock_dim} + {self.indicator_dim}×{self.stock_dim} = {self.state_dim()}

2. Field order:
   - s[0]: cash
   - s[1:{1+self.stock_dim}]: close prices (asset-major)
   - s[{1+self.stock_dim}:{1+2*self.stock_dim}]: holdings (asset-major)
   - s[{1+2*self.stock_dim}:]: indicators (INDICATOR-MAJOR, not asset-major)

3. Critical differences from generic schema:
   - NO open/high/low/volume in the state
   - Indicators are in indicator-major order: [ind1_1, ind1_2, ..., ind2_1, ind2_2, ...]
   - Close prices and holdings are separate blocks

4. Index access rules:
   - DO NOT hard-code indices beyond s[{self.state_dim()-1}]
   - DO NOT assume OHLCV structure
   - DO use {self.stock_dim} (stock_dim) and {self.indicator_dim} (indicator_dim) in loops
   - DO validate all indices are < {self.state_dim()}

Example safe access:
   close_idxs = [1 + i for i in range({self.stock_dim})]
   holding_idxs = [1 + {self.stock_dim} + i for i in range({self.stock_dim})]
   sma_idxs = [1 + 2*{self.stock_dim} + 0*{self.stock_dim} + i for i in range({self.stock_dim})]
"""

    def validate_indices(self, code: str) -> Tuple[bool, str]:
        """验证代码中的索引"""
        max_dim = self.state_dim()

        # 检查硬编码的越界索引
        import re

        # 查找所有 s[数字] 模式
        patterns = re.findall(r's\[(\d+)\]', code)
        for pattern in patterns:
            idx = int(pattern)
            if idx >= max_dim:
                return False, f"Index out of bounds: s[{idx}] >= state_dim ({max_dim})"

        # 检查循环中的索引
        # 例如: for i in range(10): s[i]
        loop_patterns = re.findall(r'for\s+\w+\s+in\s+range\((\d+)\)', code)
        for pattern in loop_patterns:
            limit = int(pattern)
            if limit > max_dim:
                return False, f"Loop range exceeds state_dim: range({limit}) > {max_dim}"

        return True, "valid"


# 在 pipeline 中使用
def validate_candidate_with_contract(
    code: str,
    contract: StateContract
) -> Tuple[bool, str]:
    """使用状态契约验证候选代码"""

    # 1. 验证索引
    valid, msg = contract.validate_indices(code)
    if not valid:
        return False, f"Index validation failed: {msg}"

    # 2. 验证必需函数
    if "def revise_state" not in code:
        return False, "Missing revise_state function"

    if "def intrinsic_reward" not in code:
        return False, "Missing intrinsic_reward function"

    # 3. 验证维度计算
    if "contract.state_dim()" in code or "state_dim" in code:
        # 候选代码应该使用正确的 state_dim
        pass  # 可以进一步验证

    return True, "valid"
```

#### 8.2.2 增强 Intrinsic Reward 验证

```python
# 增强 intrinsic_reward 的 raw-state fallback 验证

def validate_intrinsic_reward_raw_fallback(
    code: str,
    contract: StateContract,
    n_samples: int = 100,
) -> Tuple[bool, str]:
    """
    验证 intrinsic_reward 的 raw-state fallback

    检查:
    1. 在原始状态上能运行
    2. 返回非平凡信号 (std > threshold)
    3. 返回值在合理范围内
    4. 与投资组合决策相关
    """
    # 1. 加载代码
    module = load_code(code)

    # 2. 生成原始状态样本
    raw_states = []
    for _ in range(n_samples):
        s = np.random.randn(contract.state_dim())
        # 添加一些金融特定的结构
        s[0] = np.random.uniform(0, 100000)  # cash
        s[1:1+contract.stock_dim] = np.random.uniform(100, 200, contract.stock_dim)  # prices
        s[1+contract.stock_dim:1+2*contract.stock_dim] = np.random.uniform(-100, 100, contract.stock_dim)  # holdings
        raw_states.append(s)

    # 3. 测试 intrinsic_reward
    raw_rewards = []
    try:
        for s in raw_states:
            r = module.intrinsic_reward(s)
            raw_rewards.append(r)
    except Exception as e:
        return False, f"Runtime error on raw state: {e}"

    raw_rewards = np.array(raw_rewards)

    # 4. 检查非平凡性
    if np.std(raw_rewards) < 0.01:
        return False, f"Raw-state intrinsic reward is nearly constant: std={np.std(raw_rewards):.6f}"

    # 5. 检查范围
    if np.any(np.abs(raw_rewards) > 100):
        return False, f"Raw-state intrinsic reward exceeds [-100, 100]: min={np.min(raw_rewards):.2f}, max={np.max(raw_rewards):.2f}"

    # 6. 检查与持仓的相关性 (action-sensitive)
    # 计算不同持仓下的 reward 变化
    base_state = raw_states[0].copy()
    base_holdings = base_state[1+contract.stock_dim:1+2*contract.stock_dim]

    # 修改持仓
    test_state = base_state.copy()
    test_state[1+contract.stock_dim:1+2*contract.stock_dim] = base_holdings * 1.5  # 增加 50%

    base_reward = module.intrinsic_reward(base_state)
    test_reward = module.intrinsic_reward(test_state)

    if abs(test_reward - base_reward) < 0.01:
        return False, f"Raw-state intrinsic reward is action-insensitive: reward change={abs(test_reward - base_reward):.6f} when holdings change by 50%"

    return True, "valid"
```

#### 8.2.3 改进 Lipschitz 分析

```python
# 改进 Lipschitz 分析，使其更适合金融场景

def calculate_financial_lipschitz(
    states: np.ndarray,
    rewards: np.ndarray,
    contract: StateContract,
) -> Dict[str, np.ndarray]:
    """
    计算金融场景的 Lipschitz 常数

    返回:
    - raw_lipschitz: 原始状态维度的 Lipschitz 常数
    - revised_lipschitz: 扩展状态维度的 Lipschitz 常数 (如果有)
    - feature_group_lipschitz: 特征组的 Lipschitz 常数
    """
    raw_dim = contract.state_dim()
    revised_dim = states.shape[1]

    # 1. 原始维度的 Lipschitz
    raw_lipschitz = np.zeros(raw_dim)
    for i in range(raw_dim):
        sorted_indices = np.argsort(states[:, i])
        delta_r = np.diff(rewards[sorted_indices])
        delta_s = np.diff(states[sorted_indices, i])
        ratio = np.abs(delta_r) / (np.abs(delta_s) + 1e-6)
        raw_lipschitz[i] = np.max(ratio)

    # 2. 扩展维度的 Lipschitz
    if revised_dim > raw_dim:
        revised_lipschitz = np.zeros(revised_dim - raw_dim)
        for i in range(revised_dim - raw_dim):
            idx = raw_dim + i
            sorted_indices = np.argsort(states[:, idx])
            delta_r = np.diff(rewards[sorted_indices])
            delta_s = np.diff(states[sorted_indices, idx])
            ratio = np.abs(delta_r) / (np.abs(delta_s) + 1e-6)
            revised_lipschitz[i] = np.max(ratio)
    else:
        revised_lipschitz = np.array([])

    # 3. 特征组 Lipschitz (如果有特征组定义)
    feature_group_lipschitz = {}
    if hasattr(contract, 'feature_groups'):
        for group_name, indices in contract.feature_groups.items():
            group_lipschitz = []
            for idx in indices:
                if idx < raw_dim:
                    group_lipschitz.append(raw_lipschitz[idx])
                else:
                    # 扩展维度的特征
                    revised_idx = idx - raw_dim
                    if revised_idx < len(revised_lipschitz):
                        group_lipschitz.append(revised_lipschitz[revised_idx])

            feature_group_lipschitz[group_name] = np.mean(group_lipschitz)

    return {
        "raw_lipschitz": raw_lipschitz,
        "revised_lipschitz": revised_lipschitz,
        "feature_group_lipschitz": feature_group_lipschitz,
    }


def generate_lipschitz_feedback(
    lipschitz_data: Dict[str, np.ndarray],
    contract: StateContract,
    top_k: int = 5,
) -> str:
    """
    生成 Lipschitz 反馈文本
    """
    raw_lipschitz = lipschitz_data["raw_lipschitz"]
    revised_lipschitz = lipschitz_data["revised_lipschitz"]

    # 1. 最重要的原始维度
    top_raw_indices = np.argsort(raw_lipschitz)[-top_k:][::-1]

    # 2. 最重要的扩展维度
    if len(revised_lipschitz) > 0:
        top_revised_indices = np.argsort(revised_lipschitz)[-top_k:][::-1]
    else:
        top_revised_indices = []

    # 3. 生成反馈
    feedback = f"""
**Lipschitz Constants Analysis**:

Lipschitz constant measures the smoothness of the reward function with respect
to each state dimension. Lower values indicate smoother mapping.

**Most Important Raw State Dimensions** (Top {top_k}):
"""

    for idx in top_raw_indices:
        dim_name = contract.get_dim_name(idx)  # 需要实现
        feedback += f"- s[{idx}] ({dim_name}): {raw_lipschitz[idx]:.4f}\n"

    if len(top_revised_indices) > 0:
        feedback += f"\n**Most Important Revised Dimensions** (Top {top_k}):\n"
        for rank, idx in enumerate(top_revised_indices):
            revised_idx = contract.state_dim() + idx
            feedback += f"- s[{revised_idx}] (revised_{idx}): {revised_lipschitz[idx]:.4f}\n"

    # 4. 特征组分析
    if "feature_group_lipschitz" in lipschitz_data:
        feedback += "\n**Feature Group Analysis**:\n"
        for group_name, value in lipschitz_data["feature_group_lipschitz"].items():
            feedback += f"- {group_name}: {value:.4f}\n"

    return feedback
```

#### 8.2.4 改进场景族采样

```python
# 改进场景族采样，使其更明确和多样化

class ScenarioFamily:
    """场景族基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def get_prompt_instruction(self) -> str:
        """返回场景族的 Prompt 指令"""
        raise NotImplementedError

    def validate_alignment(self, code: str) -> Tuple[bool, str]:
        """验证代码是否符合场景族"""
        raise NotImplementedError


class TrendFollowScenario(ScenarioFamily):
    """趋势跟踪场景"""

    def __init__(self):
        super().__init__(
            name="trend_follow",
            description="Capture positive momentum and follow market trends"
        )

    def get_prompt_instruction(self) -> str:
        return """
**Scenario Family: Trend Following**

Focus on capturing positive momentum while managing downside risk:

Key principles:
- Reward alignment with market trend (long in uptrend, reduce in downtrend)
- Use trend strength indicators (moving averages, momentum, breakout signals)
- Reward persistence of trends (avoid whipsaw)
- Manage drawdowns during trend reversals

Suggested features for revise_state:
- Trend indicators: moving averages, momentum, breakout signals
- Trend strength: ADX, trend persistence, volatility-adjusted momentum
- Drawdown measures: recent drawdown, maximum drawdown, drawdown duration

Suggested design for intrinsic_reward:
- Positive: reward correct positioning (long when trend is up)
- Negative: penalize counter-trend positions
- Negative: penalize whipsaw (frequent position changes)
- Positive: reward trend persistence (stable trends)
"""

    def validate_alignment(self, code: str) -> Tuple[bool, str]:
        # 检查是否有趋势相关特征
        trend_keywords = ["momentum", "trend", "moving_average", "sma", "ema", "adx"]
        has_trend = any(kw in code.lower() for kw in trend_keywords)

        if not has_trend:
            return False, "trend_follow candidate should use trend indicators"

        return True, "valid"


class MeanRevertScenario(ScenarioFamily):
    """均值回归场景"""

    def __init__(self):
        super().__init__(
            name="mean_revert",
            description="Capture overbought/oversold conditions and revert to mean"
        )

    def get_prompt_instruction(self) -> str:
        return """
**Scenario Family: Mean Reversion**

Focus on capturing overbought/oversold conditions and reverting to mean:

Key principles:
- Reward contrarian positions (long when oversold, short when overbought)
- Use statistical deviation measures (Z-score, Bollinger Bands, RSI)
- Reward convergence to mean
- Manage risk during persistent trends

Suggested features for revise_state:
- Deviation measures: Z-score, Bollinger Band position, RSI
- Mean estimates: moving average, historical mean, median price
- Overbought/oversold indicators: RSI > 70, RSI < 30
- Volatility-adjusted deviation: deviation / volatility

Suggested design for intrinsic_reward:
- Positive: reward contrarian positions (opposite to recent momentum)
- Positive: reward convergence to mean
- Negative: penalize trend-chasing (aligning with recent momentum)
- Negative: penalize extreme deviations
"""

    def validate_alignment(self, code: str) -> Tuple[bool, str]:
        # 检查是否有均值回归相关特征
        revert_keywords = ["mean", "revert", "deviation", "z_score", "bollinger", "rsi"]
        has_revert = any(kw in code.lower() for kw in revert_keywords)

        if not has_revert:
            return False, "mean_revert candidate should use deviation/mean-reversion indicators"

        return True, "valid"


class RiskShieldScenario(ScenarioFamily):
    """风险屏蔽场景"""

    def __init__(self):
        super().__init__(
            name="risk_shield",
            description="Focus on downside protection and risk management"
        )

    def get_prompt_instruction(self) -> str:
        return """
**Scenario Family: Risk Shield**

Focus on downside protection and risk management:

Key principles:
- Limit downside exposure and drawdowns
- Reward diversification and low concentration
- Use volatility and drawdown measures
- Reward defensive positioning in high volatility

Suggested features for revise_state:
- Risk measures: VaR, CVaR, maximum drawdown, volatility
- Concentration: Herfindahl index, max position size
- Correlation: portfolio correlation with market
- Drawdown indicators: current drawdown, drawdown duration

Suggested design for intrinsic_reward:
- Negative: penalize high concentration
- Negative: penalize high volatility exposure
- Negative: penalize large drawdowns
- Positive: reward diversification (low correlation)
- Positive: reward defensive positioning (low exposure in high volatility)
"""

    def validate_alignment(self, code: str) -> Tuple[bool, str]:
        # 检查是否有风险管理相关特征
        risk_keywords = ["var", "cvar", "drawdown", "volatility", "concentration", "risk"]
        has_risk = any(kw in code.lower() for kw in risk_keywords)

        if not has_risk:
            return False, "risk_shield candidate should use risk management indicators"

        return True, "valid"


# 在 pipeline 中使用
def sample_from_scenario_family(
    family: ScenarioFamily,
    base_prompt: str,
    llm_client,
    k: int = 1,
) -> List[str]:
    """
    从特定场景族采样
    """
    # 1. 添加场景族指令
    scenario_instruction = family.get_prompt_instruction()
    prompt = base_prompt + "\n\n" + scenario_instruction

    # 2. 采样 k 个候选
    codes = []
    for _ in range(k):
        code = llm_client.chat(prompt)
        codes.append(code)

    # 3. 验证对齐
    valid_codes = []
    for code in codes:
        valid, msg = family.validate_alignment(code)
        if valid:
            valid_codes.append(code)
        else:
            logger.warning(f"Scenario alignment validation failed: {msg}")

    return valid_codes
```

---

## 总结

### 核心发现

1. **架构层面**:
   - ✅ 已实现 native state contract
   - ❌ 但 prompt/CoT/诊断链条仍有 generic schema 残留
   - ❌ System prompt 缺少 backend-specific 绑定

2. **Intrinsic Reward 设计**:
   - ❌ **核心问题**: G2 mode 的 raw-state fallback 不足
   - ❌ 大量候选的 intrinsic_reward 在原始状态上接近常数
   - ❌ Action-insensitive 设计问题
   - ❌ 仍把 intrinsic_reward 当作"探索奖励"，而非"风险感知引导"

3. **迭代优化**:
   - ⚠️  历史压缩可能丢失早期成功模式
   - ⚠️  场景族定义过于粗糙
   - ⚠️  算法特定适配不足

4. **从机器人到金融的思维转换**:
   - ❌ **根本问题**: LESR 的"机器人思维"未完全转换到"金融思维"
   - ❌ 把 intrinsic_reward 当作"探索奖励"，而非"风险感知引导"
   - ❌ 把 revise_state 当作"特征增强"，而非"风险暴露"
   - ❌ 把 Lipschitz 当作"平滑性"，而非"风险-收益权衡稳定性"

### 改进优先级

**P0 (立即修复)**:
1. 修复 state_contract 不一致 (build_cot_prompt 的 source_dim)
2. 重新设计 Prompt: 从"探索奖励"到"风险感知引导"
3. 增强 raw-state fallback 验证

**P1 (高优先级)**:
4. System prompt 添加 backend-specific 绑定
5. 算法特定适配 (TD3/PPO/SAC/A2C)
6. 改进场景族采样

**P2 (中优先级)**:
7. 改进历史压缩 (保留成功模式)
8. 增强 Lipschitz 分析 (特征组)
9. 添加场景族验证

### 关键洞察

**把 LESR 从机器人搬到金融，不是简单的"换个 Prompt"，而是需要重新设计核心逻辑**:

```
机器人控制:
- 状态: 物理量 (位置、速度、力矩)
- 目标: 完成任务 (前进、抓取)
- 奖励: 任务完成度 + 探索奖励
- Intrinsic: 鼓励探索新状态

金融 DRL:
- 状态: 金融量 (价格、持仓、风险)
- 目标: 风险调整收益 (Sharpe Ratio)
- 奖励: 投资收益 + 风险惩罚
- Intrinsic: 引导风险-aware 决策

⚠️  这不是"换个 Prompt"能解决的问题，而是需要重新设计 Objective！
```

---

**文档版本**: v1.0
**最后更新**: 2026-04-02
**作者**: LESR 金融交易项目分析
**相关文档**:
- [LESR系统架构分析.md](../../参考项目梳理/LESR/LESR系统架构分析.md)
- [LLM代码生成机制详解.md](../../参考项目梳理/LESR/LLM代码生成机制详解.md)
