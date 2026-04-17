# Phase 3: LESR Core Improvements - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-15
**Phase:** 03-lesr-core-improvements
**Areas discussed:** Prompt 设计策略, 质量控制与特征筛选, COT 反馈机制, 特征稳定性评估

---

## Prompt 设计策略

### 市场状态信息

| Option | Description | Selected |
|--------|-------------|----------|
| 市场感知型 | 注入训练集统计摘要（波动率、趋势强度、平均成交量、日均收益率） | ✓ |
| 通用指导型 | 仅提供特征库描述和迭代历史，不注入实时市场数据 | |
| 渐进式注入 | 初始不提供，后续迭代逐渐注入统计信息 | |

**User's choice:** 市场感知型
**Notes:** 注入基础统计摘要即可，不需要多窗口或因子预评估信息

### 迭代上下文

| Option | Description | Selected |
|--------|-------------|----------|
| 完整历史 | 所有历史迭代的完整信息（~8k tokens/迭代） | |
| 精选上下文 | 仅上一轮+历史最优组合（~2k tokens/迭代） | ✓ |
| 摘要式 | 只告诉历史最优和最差组合 | |

**User's choice:** 精选上下文

### 特征选择自由度

| Option | Description | Selected |
|--------|-------------|----------|
| 开放选择 | 从完整库中自由选择任意组合 | |
| 主题包引导 | 在预设主题包中选择，可跨包组合 | ✓ |
| 分层选择 | 先判断市场类型，系统推荐指标，LLM 在推荐范围内调整 | |

**User's choice:** 主题包引导

### 主题包结构

| Option | Description | Selected |
|--------|-------------|----------|
| 4包固定结构 | 趋势、波动率、均值回归、成交量各一个包 | ✓ |
| 动态包结构 | 根据股票特性动态调整包内容 | |
| 软约束 | 不设包但要求覆盖至少2个类别 | |

**User's choice:** 4包固定结构 — 趋势(RSI,MACD,EMA_Cross,Momentum)、波动率(Bollinger,ATR,Volatility)、均值回归(Stochastic,Williams_%R,CCI)、成交量(OBV,Volume_Ratio,ADX)

### 市场统计内容

| Option | Description | Selected |
|--------|-------------|----------|
| 基础统计摘要 | 波动率、趋势强度、平均成交量、日均收益率（~100 tokens） | ✓ |
| 多窗口统计 | 加上不同时间窗口的统计 | |
| 含因子预评估 | 附上各指标在训练集上的 IC/IR 值 | |

**User's choice:** 基础统计摘要

---

## 质量控制与特征筛选

### 质量守门严格度

| Option | Description | Selected |
|--------|-------------|----------|
| 静态验证 | JSON解析+指标存在+参数范围+数量检查 | |
| 静态+因子预评估 | 静态验证+用训练数据计算IC/方差，拒绝IC≈0或方差≈0 | ✓ |
| 全流程验证 | 预评估+特征间相关性检查+去冗余 | |

**User's choice:** 静态+因子预评估

### 特征数量

| Option | Description | Selected |
|--------|-------------|----------|
| 固定5-10个 | 与ROADMAP成功标准一致 | ✓ |
| 动态上限 | 根据特征库大小调整 | |
| 提议-筛选模式 | LLM自由提议，系统保留IC最高的K个 | |

**User's choice:** 固定5-10个

### 筛选标准

| Option | Description | Selected |
|--------|-------------|----------|
| IC+方差筛选 | IC>阈值 + 方差>阈值，简单可解释 | ✓ |
| IC+去冗余 | IC>阈值 + 与已有特征相关性<0.9 | |
| IC+SHAP排序 | IC筛选后用SHAP重要性排序保留top-K | |

**User's choice:** IC+方差筛选

### 阈值严格度

| Option | Description | Selected |
|--------|-------------|----------|
| 宽松阈值 | IC>0.02, variance>1e-6 | ✓ |
| 严格阈值 | IC>0.05, variance>1e-4 | |
| 自适应阈值 | 根据数据统计分布自动确定 | |

**User's choice:** 宽松阈值

### 同类型指标冲突

| Option | Description | Selected |
|--------|-------------|----------|
| 保留强者 | 同类指标保留IC更高的那个 | ✓ |
| 允许共存 | 两者都保留 | |
| 自动融合 | 自动创建差值或比值新特征 | |

**User's choice:** 保留强者

---

## COT 反馈机制

### 反馈信息级别

| Option | Description | Selected |
|--------|-------------|----------|
| 策略绩效+因子评估 | Sharpe/MaxDD/TotalReturn + 每个特征的IC/IR | ✓ |
| 仅策略绩效 | 只有策略级指标 | |
| 全信息（含最差交易） | 策略+因子+最差交易分析 | |

**User's choice:** 策略绩效+因子评估

### 负面指导具体度

| Option | Description | Selected |
|--------|-------------|----------|
| 具体拒绝原因 | 列出被拒绝特征及原因+总结性"不要做X"指导 | ✓ |
| 模糊引导 | 只说"上次趋势类指标表现不佳" | |
| 仅正面反馈 | 只展示成功组合 | |

**User's choice:** 具体拒绝原因

### 反馈时机

| Option | Description | Selected |
|--------|-------------|----------|
| 即时反馈 | 每个候选评估完立即生成 | |
| 批量反馈 | 所有候选评估完后统一生成 | ✓ |

**User's choice:** 批量反馈

---

## 特征稳定性评估

### 时间段划分

| Option | Description | Selected |
|--------|-------------|----------|
| 固定子期间划分 | 训练集分3-4等长期间，计算每个子期间IC | ✓ |
| 滚动IC序列 | 用滚动窗口(20天)计算IC序列 | |
| 按市场状态分层 | 用regime_detector标记后分层计算IC | |

**User's choice:** 固定子期间划分

### 稳定性标准

| Option | Description | Selected |
|--------|-------------|----------|
| IC均值+波动比 | IC均值>阈值 且 IC标准差<2*IC均值 | ✓ |
| 全期间正IC | IC在所有子期间都为正 | |
| IC Sharpe Ratio | IC_mean/IC_std > 0.5 | |

**User's choice:** IC均值+波动比

### 不稳定特征处理

| Option | Description | Selected |
|--------|-------------|----------|
| 直接移除 | 从特征集移除，在COT反馈中告知LLM | ✓ |
| 降权保留 | DQN训练时降低权重 | |
| 标记并反馈 | 不自动处理，告知LLM让其自己决定 | |

**User's choice:** 直接移除

---

## Claude's Discretion

- 具体 prompt 模板措辞和格式
- 主题包中每个指标的默认参数范围
- 特征库的具体实现（函数签名、输出归一化方式）
- 报告格式（markdown tables vs LaTeX）
- IC 阈值的具体数值微调

## Deferred Ideas

- Custom feature proposal by LLM (outside the library) — future phase
- Adaptive feature library (growing over iterations) — v2
- Rolling IC-based stability — more granular but higher compute
- Market regime-stratified stability — could add later
