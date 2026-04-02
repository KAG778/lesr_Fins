# FINSABER 项目 LLM 提示词设计方案

## 一、项目背景与任务分析

### 1.1 FINSABER vs LESR 原始任务对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINSABER 与 LESR 原始任务对比                              │
└─────────────────────────────────────────────────────────────────────────────┘

维度              │ LESR (MuJoCo)        │ FINSABER (金融交易)
─────────────────┼──────────────────────┼────────────────────────────────
任务类型          │ 连续控制             │ 序列决策
                  │ (运动/导航)          │ (交易策略)
─────────────────┼──────────────────────┼────────────────────────────────
状态空间          │ 关节角度、角速度      │ 价格序列、成交量、技术指标
                  │ 物理量 (rad, m/s)    │ 金融量 (价格, 成交量)
─────────────────┼──────────────────────┼────────────────────────────────
目标              │ 最大化前向速度        │ 最大化夏普比率
                  │ 到达目标位置          │ 最小化回撤
─────────────────┼──────────────────────┼────────────────────────────────
约束              │ 物理定律              │ 市场规律
                  │ (能量守恒、摩擦力)    │ (资金限制、交易成本)
─────────────────┼──────────────────────┼────────────────────────────────
风险              │ 摔倒、碰撞            │ 资金损失、过度拟合
                  │ 物理损伤              │ 策略失效
─────────────────┼──────────────────────┼────────────────────────────────
领域知识          │ 经典力学、控制理论    │ 技术分析、量化金融
                  │ 步态模式              │ 市场微观结构
─────────────────┼──────────────────────┼────────────────────────────────
性能指标          │ 累积奖励              │ 夏普比率、总收益
                  │ 成功率                │ 最大回撤、胜率
─────────────────┼──────────────────────┼────────────────────────────────
数据特性          │ 连续、平滑            │ 噪声、非平稳
                  │ 低维度                │ 高维度、时序
─────────────────┼──────────────────────┼────────────────────────────────
```

### 1.2 关键差异分析

#### 差异 1：状态空间语义

```python
# ========== LESR: 物理状态 (清晰、直观) ==========
"""
HalfCheetah 状态空间:
- s[0]: z-coordinate of the front tip (position in meters)
- s[1]: angle of the front tip (angle in radians)
- s[8]: x-coordinate velocity (velocity in m/s) ← 前向速度！

语义清晰：
- 单位明确 (m, rad, m/s)
- 物理意义直观 (位置、速度、角度)
- 因果关系明确 (力 → 加速度 → 速度 → 位置)
"""

# ========== FINSABER: 金融状态 (抽象、复杂) ==========
"""
FINSABER 状态空间:
- s[0:19]: 20天收盘价序列 (价格 in dollars)
- s[20:39]: 20天成交量序列 (成交量 in shares)
- s[40:59]: 可能的技术指标 (MA, RSI, MACD...)

语义复杂：
- 单位相同但尺度差异大 (价格、成交量)
- 物理意义抽象 (价格 → 趋势、动量)
- 因果关系模糊 (价格变化受多重因素影响)
- 时序依赖性强 (历史价格影响未来)
"""
```

#### 差异 2：优化目标

```python
# ========== LESR: 单一目标 ==========
"""
目标: 最大化累积奖励
  - HalfCheetah: reward = forward_velocity - energy_cost
  - AntMaze: reward = 1.0 if reached_goal else 0.0

特点:
  - 单一目标明确
  - 奖励函数简单
  - 优化方向清晰
"""

# ========== FINSABER: 多目标权衡 ==========
"""
目标: 最大化风险调整后收益
  - 夏普比率 = (收益 - 无风险利率) / 波动率
  - 总收益 = 最终财富 / 初始财富
  - 最大回撤 = max((峰值 - 当前值) / 峰值)
  - 胜率 = 盈利交易次数 / 总交易次数

特点:
  - 多目标权衡 (收益 vs 风险)
  - 奖励函数复杂
  - 优化方向不唯一
  - 需要考虑交易成本
"""
```

#### 差异 3：领域知识

```python
# ========== LESR: 经典力学 ==========
"""
物理定律:
  - 牛顿第二定律: F = ma
  - 能量守恒: KE + PE = constant
  - 摩擦力: f = μN

应用:
  - 前向速度 = ∫(加速度)dt
  - 动能 = 0.5 * m * v^2
  - 功率 = torque * angular_velocity

特点:
  - 普适性强
  - 计算精确
  - 因果明确
"""

# ========== FINSABER: 量化金融 ==========
"""
市场规律:
  - 趋势: 价格持续朝一个方向运动
  - 均值回归: 价格偏离均值后回归
  - 动量: 强者恒强
  - 波动率聚集: 大波动后跟随大波动

技术指标:
  - 趋势: SMA, EMA, MACD
  - 动量: RSI, ROC, 随机指标
  - 波动率: 标准差, ATR, 布林带
  - 成交量: OBV, 成交量加权

特点:
  - 经验性强
  - 市场依赖
  - 噪声大
  - 不确定性高
"""
```

### 1.3 FINSABER 任务分类

根据LESR的任务分类框架，FINSABER属于**新的任务类型**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINSABER 任务特征分析                                      │
└─────────────────────────────────────────────────────────────────────────────┘

任务类型: 序列决策任务 (Sequential Decision Making)
子类型: 金融交易 (Financial Trading)

关键特征:
✅ 时序依赖: 历史价格影响未来决策
✅ 非平稳性: 市场环境随时间变化
✅ 高噪声: 价格波动包含大量随机成分
✅ 风险敏感: 需要考虑资金管理
✅ 延迟反馈: 交易决策的效果需要时间体现
✅ 部分可观测: 只能看到公开市场信息

与LESR任务类型的对比:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│              │ 运动任务     │ 导航任务     │ FINSABER     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 状态空间      │ 连续、平滑   │ 连续、平滑   │ 离散、噪声   │
│ 目标明确性    │ 高           │ 高           │ 中           │
│ 领域知识      │ 经典力学     │ 经典力学     │ 量化金融     │
│ 风险类型      │ 物理损伤     │ 物理损伤     │ 资金损失     │
│ 数据特性      │ 低维、确定性 │ 中维、确定性 │ 高维、随机性 │
│ 反馈延迟      │ 即时         │ 短延迟       │ 长延迟       │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

## 二、FINSABER 提示词设计原则

### 2.1 核心设计原则

基于LESR的提示词设计三角原则，FINSABER需要调整：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  FINSABER 提示词设计三角原则                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                ┌───────────────────┐
                │   金融语义丰富性   │
                │ (Financial Semantics)│
                │                   │
                │ - 技术指标含义     │
                │ - 市场规律知识     │
                │ - 风险管理意识     │
                └─────────┬─────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────┴───────┐         │         ┌───────┴───────┐
│  金融约束明确  │─────────┼─────────│  可执行验证   │
│ (Financial     │         │         │ (Executable   │
│  Constraints)  │         │         │  Validation)  │
│               │         │         │               │
│ - 函数签名    │         │         │ - 数值稳定性  │
│ - 输入输出    │         │         │ - 边界处理    │
│ - 风险限制    │         │         │ - 代码可运行  │
│ - 交易成本    │         │         │               │
└───────────────┘         │         └───────────────┘
                          │
                ┌─────────┴─────────┐
                │   迭代反馈闭环     │
                │ (Iterative Feedback)│
                │                   │
                │ - 绩效指标分析     │
                │ - 特征重要性       │
                │ - 风险归因分析     │
                │ - 市场环境归因     │
                └───────────────────┘
```

### 2.2 三大原则详解

#### 原则 1：金融语义丰富性 (Financial Semantics)

```python
# ========== 为什么需要金融语义？ ==========
"""
问题: LLM 不理解金融数据的含义

错误示例:
```python
def revise_state(prices):
    # LLM 不知道价格意味着什么
    return prices * 2  # 无意义的变换
```

正确示例:
```python
def revise_state(prices):
    # 明确金融语义
    returns = np.diff(prices) / prices[:-1]  # 收益率
    momentum = np.mean(returns[-5:])         # 动量 (5日平均收益)
    volatility = np.std(returns[-20:])       # 波动率 (20日标准差)
    return np.concatenate([prices, [momentum, volatility]])
```

关键: 在Prompt中明确金融术语的含义
"""
```

#### 原则 2：金融约束明确 (Financial Constraints)

```python
# ========== 为什么需要金融约束？ ==========
"""
问题: 金融交易有独特的约束

约束 1: 资金约束
  - 不能交易超过可用资金的量
  - 需要考虑仓位管理

约束 2: 交易成本
  - 佣金: commission * 交易金额
  - 滑点: 市价交易的执行偏差
  - 印花税: 某些市场的税费

约束 3: 风险约束
  - 单笔交易风险 < 总资金的 X%
  - 最大回撤 < Y%
  - 波动率 < Z%

约束 4: 市场约束
  - 交易时间: 只能在开盘时间交易
  - 涨跌停: 价格限制
  - 流动性: 成交量限制

在Prompt中明确这些约束，避免LLM生成不切实际的策略。
"""
```

#### 原则 3：迭代反馈闭环 (Iterative Feedback)

```python
# ========== 为什么需要特定的反馈机制？ ==========
"""
问题: 金融策略的反馈不同于物理任务

差异 1: 性能指标
  - LESR: 累积奖励、成功率
  - FINSABER: 夏普比率、最大回撤、胜率

差异 2: 特征重要性
  - LESR: Lipschitz 常数 (平滑性)
  - FINSABER: 相关系数、SHAP值、信息增益

差异 3: 失败模式
  - LESR: 摔倒、震荡、局部最优
  - FINSABER: 过拟合、过度交易、风险失控

差异 4: 市场环境
  - LESR: 固定环境
  - FINSABER: 牛市、熊市、震荡市

需要设计专门的反馈Prompt来分析这些金融特有的问题。
"""
```

### 2.3 提示词设计检查清单

```python
# ========== FINSABER 提示词设计检查清单 ==========

def finsaber_prompt_checklist(prompt_template):
    """
    FINSABER 提示词设计检查清单
    """
    print("=== FINSABER 提示词设计检查清单 ===\n")

    checks = []

    # 1. 金融语义丰富性
    checks.append(("包含金融术语定义",
                  any(term in prompt_template.lower() for term in
                      ['return', 'volatility', 'momentum', 'trend', 'indicator'])))

    checks.append(("包含技术指标说明",
                  'SMA' in prompt_template or 'EMA' in prompt_template or
                  'RSI' in prompt_template or 'MACD' in prompt_template))

    checks.append(("包含风险管理提示",
                  'risk' in prompt_template.lower() or
                  'drawdown' in prompt_template.lower()))

    # 2. 金融约束明确
    checks.append(("包含交易成本说明",
                  'commission' in prompt_template.lower() or
                  'cost' in prompt_template.lower()))

    checks.append(("包含数值范围限制",
                  '[-100, 100]' in prompt_template or 'range' in prompt_template.lower()))

    checks.append(("包含边界处理提示",
                  'divide' in prompt_template.lower() or 'zero' in prompt_template.lower()))

    # 3. 可执行验证
    checks.append(("包含函数签名",
                  'def revise_state' in prompt_template))

    checks.append(("包含示例代码",
                  '```python' in prompt_template))

    checks.append(("包含numpy使用提示",
                  'numpy' in prompt_template.lower() or 'np.' in prompt_template))

    # 4. 迭代反馈
    checks.append(("包含绩效指标说明",
                  any(metric in prompt_template.lower() for metric in
                      ['sharpe', 'return', 'drawdown', 'win rate'])))

    # 5. 金融特有
    checks.append(("包含时序处理提示",
                  'time' in prompt_template.lower() or
                  'sequence' in prompt_template.lower() or
                  'history' in prompt_template.lower()))

    checks.append(("包含过拟合警告",
                  'overfit' in prompt_template.lower() or
                  'generaliz' in prompt_template.lower()))

    # 输出结果
    pass_count = sum(1 for _, result in checks if result)
    total_count = len(checks)

    for check, result in checks:
        print(f"{'✅' if result else '❌'} {check}")

    print(f"\n通过率: {pass_count}/{total_count} ({pass_count/total_count*100:.1f}%)")

    if pass_count == total_count:
        print("🎉 所有检查通过！提示词设计完善。")
    elif pass_count >= total_count * 0.8:
        print("⚠️  大部分检查通过，建议优化缺失项。")
    else:
        print("❌ 检查未通过，提示词需要大幅改进。")

    return checks

# ========== 示例运行 ==========
# finsaber_prompt_checklist(finsaber_prompt_template)
```

## 三、FINSABER 初始化提示词设计

### 3.1 完整的初始化提示词模板

```python
# ========== FINSABER 初始化提示词模板 ==========

FINSABER_INIT_PROMPT_TEMPLATE = """
你是金融交易领域的量化分析专家和特征工程专家。你的任务是为股票交易策略生成状态表示函数和内在奖励函数。

## 任务背景

我们将使用强化学习来训练股票交易策略。策略在每个交易日需要决定是否买入、卖出或持有股票。为了帮助强化学习算法更好地学习，我们需要设计合适的状态表示和内在奖励。

## 可用原始数据

### 1. 价格数据（最近20个交易日）

对于每个交易日 t，我们有以下数据（按时间顺序，从最早到最新）：

- 原始状态数组格式（120维）:
  ```
  s[0:19]:   收盘价 (Close Price) - 单位: 美元 ($)
  s[20:39]:  开盘价 (Open Price) - 单位: 美元 ($)
  s[40:59]:  最高价 (High Price) - 单位: 美元 ($)
  s[60:79]:  最低价 (Low Price) - 单位: 美元 ($)
  s[80:99]:  成交量 (Volume) - 单位: 股数 (shares)
  s[100:119]: 调整后收盘价 (Adjusted Close) - 单位: 美元 ($)
  ```

### 2. 数据特性说明

**重要金融概念**:
- **收盘价 (Close)**: 每个交易日最后交易的价格，是最常用的价格数据
- **开盘价 (Open)**: 每个交易日第一笔交易的价格
- **最高/最低价 (High/Low)**: 当日交易的最高和最低价格，反映日内波动
- **成交量 (Volume)**: 当日交易的股票数量，反映市场活跃度
- **调整后收盘价 (Adj Close)**: 考虑分红和拆股后的收盘价，用于计算真实收益

**关键洞察**:
- 价格数据是非平稳的（随时间变化），通常使用**收益率**而非原始价格
- 收益率 = (当前价格 - 过去价格) / 过去价格
- 成交量通常需要**对数变换**或**标准化**
- 不同股票的价格尺度差异很大，需要考虑归一化

## 任务要求

请生成两个Python函数：

### 函数 1: `revise_state(raw_state)`

**输入**:
- `raw_state`: numpy数组，形状(120,)，包含20天的原始OHLCV数据

**输出**:
- `enhanced_state`: numpy数组，形状(120 + k,)，其中k是你添加的特征数量
  - 前120维保留原始数据
  - 后k维是你计算的金融特征

**建议的特征类别**:

#### 类别 1: 趋势指标 (Trend Indicators)
趋势反映价格的运动方向，是交易决策的基础。

推荐特征:
```python
# 价格动量 (Price Momentum)
momentum_5d = (close[-1] - close[-6]) / close[-6]  # 5日收益率
momentum_10d = (close[-1] - close[-11]) / close[-11]  # 10日收益率
momentum_20d = (close[-1] - close[-21]) / close[-21]  # 20日收益率

# 移动平均线 (Moving Average)
ma5 = np.mean(close[-5:])    # 5日均线
ma10 = np.mean(close[-10:])  # 10日均线
ma20 = np.mean(close[-20:])  # 20日均线

# 趋势强度
trend_strength = (ma5 - ma20) / ma20  # 短期均线相对长期均线
```

#### 类别 2: 动量指标 (Momentum Indicators)
动量反映价格变化的速度和力度。

推荐特征:
```python
# RSI (Relative Strength Index) - 相对强弱指标
# RSI 衡量价格变动的速度和变化，范围[0, 100]
# RSI > 70: 超买，价格可能回调
# RSI < 30: 超卖，价格可能反弹
delta = np.diff(close)
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = np.mean(gain[-14:])
avg_loss = np.mean(loss[-14:])
rs = avg_gain / (avg_loss + 1e-6)
rsi = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
ema12 = np.mean(close[-12:])  # 简化版
ema26 = np.mean(close[-26:])
macd = ema12 - ema26
```

#### 类别 3: 波动率指标 (Volatility Indicators)
波动率反映价格的不确定性，是风险的重要度量。

推荐特征:
```python
# 历史波动率 (Historical Volatility)
returns = np.diff(close) / close[:-1]
volatility_10d = np.std(returns[-10:])  # 10日波动率
volatility_20d = np.std(returns[-20:])  # 20日波动率

# ATR (Average True Range) - 平均真实波幅
# ATR 考虑了跳空情况，更真实地反映波动
tr = np.maximum(high - low,
                np.maximum(abs(high - close_prev),
                          abs(low - close_prev)))
atr = np.mean(tr[-14:])

# 布林带宽度 (Bollinger Bands Width)
# 布林带宽度反映波动率的变化
bb_upper = ma20 + 2 * volatility_20d
bb_lower = ma20 - 2 * volatility_20d
bb_width = (bb_upper - bb_lower) / ma20
```

#### 类别 4: 成交量指标 (Volume Indicators)
成交量反映市场参与度和价格变动的可信度。

推荐特征:
```python
# 成交量变化率
volume_change = (volume[-1] - volume[-6]) / volume[-6]

# 成交量移动平均
volume_ma5 = np.mean(volume[-5:])
volume_ma20 = np.mean(volume[-20:])

# 价量关系 (Price-Volume Relation)
# 成交量放大 + 价格上涨 = 上涨趋势确认
price_change = (close[-1] - close[-2]) / close[-2]
volume_price_trend = price_change * (volume[-1] / volume_ma20)

# OBV (On-Balance Volume) - 能量潮
obv = np.sum(np.sign(np.diff(close)) * volume[1:])
```

#### 类别 5: 市场微观结构指标
反映市场更深层次的信息。

推荐特征:
```python
# 日内波动幅度
daily_range = (high[-1] - low[-1]) / close[-1]

# 跳空幅度 (Gap)
gap = (open[-1] - close[-2]) / close[-2]

# 买卖压力 (Buying/Selling Pressure)
# 如果收盘价接近最高价，说明买方力量强
pressure = (close[-1] - low[-1]) / (high[-1] - low[-1] + 1e-6)
```

### 函数 2: `intrinsic_reward(enhanced_state)`

**输入**:
- `enhanced_state`: revise_state的输出，包含原始数据和计算特征

**输出**:
- `reward_value`: 标量值，范围[-100, 100]
  - 正值表示有利于交易的状态
  - 负值表示不利于交易的状态
  - 数值大小表示有利/不利的程度

**设计原则**:

#### 原则 1: 趋势跟随
```python
# 如果处于上升趋势，给予正奖励
# 如果处于下降趋势，给予负奖励
if momentum_5d > 0 and momentum_10d > 0:
    reward = 10.0 * momentum_5d  # 上升趋势，正奖励
elif momentum_5d < 0 and momentum_10d < 0:
    reward = -10.0 * abs(momentum_5d)  # 下降趋势，负奖励
```

#### 原则 2: 风险调整
```python
# 考虑波动率，高风险状态给予惩罚
volatility_penalty = -5.0 * volatility_10d
reward += volatility_penalty
```

#### 原则 3: 极端值惩罚
```python
# RSI极端值（超买超卖）给予惩罚
if rsi > 70 or rsi < 30:
    reward -= 5.0
```

#### 原则 4: 成交量确认
```python
# 成交量放大的趋势更可靠
if volume_price_trend > 0:
    reward += 2.0  # 价量配合，额外奖励
```

## 约束条件

### 1. 数值稳定性
```python
# 必须处理除零、空值等边界情况
❌ 错误: ratio = price / volume  # 可能除零
✅ 正确: ratio = price / (volume + 1e-6)  # 添加epsilon

# 必须处理NaN、Inf
❌ 错误: return np.log(price)  # price可能<=0
✅ 正确: return np.log(price + 1e-6)  # 避免log(0)
```

### 2. 特征范围
```python
# 建议将特征值限制在合理范围
# 过大的特征值会导致训练不稳定
❌ 错误: feature = price * 1000  # 可能非常大
✅ 正确: feature = np.clip(price / np.mean(prices), -10, 10)
```

### 3. 维度限制
```python
# 不要添加过多特征（避免过拟合）
# 建议: 5-15个额外特征
❌ 错误: 添加100个特征
✅ 正确: 添加10个精选特征
```

### 4. 计算效率
```python
# 避免复杂的循环计算
❌ 错误:
```python
for i in range(len(close)):
    for j in range(i+1, len(close)):
        # O(n^2)复杂度
```

✅ 正确:
```python
# 使用向量化操作
np.dot(close, weights)  # O(n)复杂度
```
```

### 5. 代码可读性
```python
# 添加清晰的注释
def revise_state(raw_state):
    # 提取价格数据
    close = raw_state[0:20]
    volume = raw_state[80:100]

    # 计算动量指标
    momentum = (close[-1] - close[-5]) / close[-5]

    # 返回增强状态
    return np.concatenate([raw_state, [momentum]])
```

## 输出格式

请返回完整可执行的Python代码，格式如下：

```python
import numpy as np

def revise_state(raw_state):
    \"\"\"
    将原始状态转换为增强状态

    Args:
        raw_state: numpy数组，形状(120,)，包含20天OHLCV数据

    Returns:
        enhanced_state: numpy数组，形状(120 + k,)，包含原始数据和计算特征
    \"\"\"
    # 提取各类数据
    close = raw_state[0:20]
    open_price = raw_state[20:40]
    high = raw_state[40:60]
    low = raw_state[60:80]
    volume = raw_state[80:100]
    adj_close = raw_state[100:120]

    # 计算特征（你的实现）
    # ...

    # 返回增强状态
    return np.concatenate([raw_state, [feature1, feature2, ...]])

def intrinsic_reward(enhanced_state):
    \"\"\"
    计算内在奖励

    Args:
        enhanced_state: numpy数组，revise_state的输出

    Returns:
        reward: 标量值，范围[-100, 100]
    \"\"\"
    # 提取特征
    # ...

    # 计算奖励
    reward = ...  # 你的实现

    # 确保在范围内
    return np.clip(reward, -100, 100)
```

## 重要提示

1. **金融领域知识**: 利用你对技术分析、量化金融的理解来设计特征
2. **简单有效**: 简单的特征组合往往优于复杂的特征工程
3. **避免过拟合**: 不要添加过多特征，5-15个为宜
4. **数值稳定**: 务必处理边界情况，确保代码健壮性
5. **可解释性**: 优先选择有明确金融含义的特征

开始生成代码：
"""
```

### 3.2 关键设计点说明

```python
# ========== 关键设计点 1: 详细的金融语义 ==========
"""
为什么这么详细？

LESR中:
- s[8]: x-coordinate velocity (m/s) ← 简单、直观

FINSABER中:
- s[0:19]: 收盘价序列 ← 需要解释什么是收盘价、为什么重要、如何使用

详细的金融语义帮助LLM:
1. 理解数据的含义
2. 选择合适的变换
3. 设计有意义的特征
4. 避免常见错误
"""

# ========== 关键设计点 2: 分类特征建议 ==========
"""
为什么要分类？

LESR中:
- 特征少，直接列举即可

FINSABER中:
- 特征多，需要分类组织
- 帮助LLM系统化思考
- 确保特征多样性

分类:
1. 趋势指标 → 方向
2. 动量指标 → 速度
3. 波动率指标 → 风险
4. 成交量指标 → 确认
5. 微观结构指标 → 深度
"""

# ========== 关键设计点 3: 具体计算示例 ==========
"""
为什么要提供代码示例？

LESR中:
- 物理计算简单 (F=ma)

FINSABER中:
- 金融计算复杂 (RSI, MACD)
- 提供示例减少错误
- 帮助LLM理解预期输出

示例的作用:
1. 展示计算方法
2. 提供最佳实践
3. 设置质量标准
"""

# ========== 关键设计点 4: 内在奖励设计原则 ==========
"""
为什么要这么多原则？

LESR中:
- reward = forward_velocity - energy_cost
- 简单、直接

FINSABER中:
- 需要权衡多个因素
- 趋势 vs 风险
- 收益 vs 成本
- 过拟合风险

原则帮助LLM:
1. 理解金融权衡
2. 设计合理的奖励
3. 避免极端行为
"""

# ========== 关键设计点 5: 明确的约束条件 ==========
"""
为什么要这么多约束？

LESR中:
- 物理约束自然清晰

FINSABER中:
- 金融约束容易被忽视
- 数值稳定性问题多
- 过拟合风险高

约束确保:
1. 代码可运行
2. 训练稳定
3. 策略可行
"""
```

## 四、FINSABER 迭代反馈提示词设计

### 4.1 COT 反馈提示词模板

```python
# ========== FINSABER COT 反馈提示词模板 ==========

FINSABER_COT_FEEDBACK_PROMPT = """
我们已成功使用 {sample_count} 个不同的状态表示函数训练了股票交易策略，每个函数都关联一个策略的回测结果。

在训练过程中，我们监控了:
1. 策略的绩效指标
   - 夏普比率 (Sharpe Ratio): 风险调整后收益
   - 总收益 (Total Return): 累积收益率
   - 最大回撤 (Max Drawdown): 最大损失幅度
   - 胜率 (Win Rate): 盈利交易占比

2. 特征重要性分析
   - 与收益的相关系数
   - 与风险的相关系数
   - 特征预测力排名

以下是详细结果:

{training_results_details}

**绩效分析**:
- 最佳策略夏普比率: {best_sharpe:.3f} (样本 #{best_id})
- 最差策略夏普比率: {worst_sharpe:.3f} (样本 #{worst_id})
- 平均夏普比率: {avg_sharpe:.3f}
- 夏普比率标准差: {std_sharpe:.3f}

**特征重要性洞察**:

最佳样本 (#{best_id}) 的特征分析:
{best_feature_analysis}

成功原因:
- {best_success_reason_1}
- {best_success_reason_2}
- {best_success_reason_3}

最差样本 (#{worst_id}) 的特征分析:
{worst_feature_analysis}

失败原因:
- {worst_failure_reason_1}
- {worst_failure_reason_2}
- {worst_failure_reason_3}

**关键发现**:

1. 高相关特征:
{high_correlation_features}

2. 低相关特征:
{low_correlation_features}

3. 特征组合模式:
{feature_combination_patterns}

4. 市场环境影响:
{market_environment_effects}

**改进建议**:

基于以上分析，请回答以下问题:

(a) 为什么最佳样本的特征组合有效？从金融理论角度分析

(b) 为什么最差样本的特征组合失败？识别关键问题

(c) 如何改进特征设计？
    - 应该保留哪些特征？
    - 应该移除哪些特征？
    - 应该尝试哪些新特征？

(d) 如何优化内在奖励函数？
    - 当前奖励设计的问题
    - 改进方向
    - 权重调整建议

(e) 如何避免过拟合？
    - 特征数量控制
    - 正则化方法
    - 样本外验证

请基于以上分析，提供改进的状态表示函数和内在奖励函数。重点关注:
- 保留并优化高相关性特征
- 简化或移除低相关性特征
- 尝试新的特征组合
- 改进内在奖励函数设计
- 确保泛化能力

开始生成改进的代码：
"""
```

### 4.2 迭代改进提示词模板

```python
# ========== FINSABER 迭代改进提示词模板 ==========

FINSABER_ITERATION_PROMPT = """
你是金融交易领域的量化分析专家。正在优化股票交易策略的状态表示（第{iteration}轮）。

## 任务回顾

策略目标: 设计最优的状态表示和内在奖励，帮助强化学习算法学习盈利的交易策略

原始数据: 20天OHLCV数据（120维）
绩效指标: 夏普比率（主要）、总收益、最大回撤、胜率

## 历史经验

{history_summary}

### 迭代 1 结果:
样本 1: 夏普比率 1.234
特征: [5日动量, 10日动量, 20日动量, RSI, MACD, 10日波动率]
成功原因: 多时间尺度动量特征捕捉趋势，RSI提供反转信号

样本 2: 夏普比率 0.856
特征: [价格, 成交量, 5日收益率, 10日收益率]
失败原因: 特征过于简单，缺少风险管理

样本 3: 夏普比率 0.654
特征: [50个技术指标]
失败原因: 特征过多导致过拟合，噪声干扰

分析结论:
- 5-10个精选特征优于大量特征
- 多时间尺度组合有效
- 需要平衡趋势跟踪和均值回归

### 迭代 2 结果:
样本 1: 夏普比率 1.456
特征: [5日动量, 10日动量, 20日动量, RSI, MACD, ATR, 布林带宽度, 成交量变化]
改进: 添加ATR和布林带宽度，更好捕捉波动率

样本 2: 夏普比率 1.123
特征: [动量, 波动率, 成交量压力]
改进: 简化特征，关注核心信号

分析结论:
- 波动率特征显著提升夏普比率
- 成交量确认信号有帮助
- 特征数量控制在8-10个最佳

### 迭代 3 结果:
{iteration_3_results}

## 当前最佳策略

最优样本 (#{best_id}):
夏普比率: {best_sharpe:.3f}
特征: {best_features}
代码:
```python
{best_code}
```

特征重要性:
{best_feature_importance}

## 当前挑战

{current_challenges}

例如:
- 牛市表现好，熊市表现差 → 需要市场环境识别
- 趋势行情盈利，震荡亏损 → 需要市场状态判断
- 过度交易 → 需要交易频率控制

## 改进方向

基于历史经验，本轮迭代重点关注:

1. **特征优化**:
   - 保留高相关性特征（相关系数 > 0.3）
   - 移除低相关性特征（相关系数 < 0.1）
   - 尝试新特征组合
   - 考虑特征交互作用

2. **内在奖励改进**:
   - 当前问题: {current_reward_issues}
   - 改进方向: {reward_improvement_direction}
   - 权重调整: {weight_adjustment_suggestions}

3. **风险管理**:
   - 波动率调整
   - 回撤控制
   - 仓位管理

4. **市场适应性**:
   - 牛熊市识别
   - 趋势震荡判断
   - 波动率 regime 切换

## 约束条件

- 特征数量: 5-15个
- 特征值范围: [-10, 10]
- 奖励范围: [-100, 100]
- 必须处理数值稳定性
- 避免过度拟合

## 任务要求

请基于以上历史经验和改进方向，生成改进的状态表示函数和内在奖励函数。

**输出格式**:
```python
import numpy as np

def revise_state(raw_state):
    \"\"\"改进的状态表示函数\"\"\"
    # 你的实现
    return enhanced_state

def intrinsic_reward(enhanced_state):
    \"\"\"改进的内在奖励函数\"\"\"
    # 你的实现
    return np.clip(reward, -100, 100)
```

**改进说明**:
在代码后简要说明你的改进思路和预期效果。

开始生成代码：
"""
```

### 4.3 反馈生成机制

```python
# ========== FINSABER 反馈生成伪代码 ==========

def generate_finsaber_feedback(training_results, raw_state_dim):
    """
    为FINSABER生成COT反馈

    关键差异（vs LESR）:
    1. 绩效指标: Sharpe/Drawdown vs Cumulative Reward
    2. 特征重要性: Correlation/SHAP vs Lipschitz
    3. 失败模式: Overfitting/Overtrading vs Falling/Oscillation
    4. 环境因素: Market Regime vs Fixed Environment
    """

    # 1. 性能排序
    sorted_results = sort_by_sharpe(training_results)
    best = sorted_results[-1]
    worst = sorted_results[0]

    # 2. 特征重要性分析（替代Lipschitz）
    feature_importance = analyze_feature_importance(
        results=training_results,
        methods=['correlation', 'shap']  # 金融特定的方法
    )

    # 3. 识别成功/失败模式
    success_patterns = identify_success_patterns(best)
    failure_patterns = identify_failure_patterns(worst)

    # 4. 市场环境分析
    market_analysis = analyze_market_regime(
        training_results,
        regimes=['bull', 'bear', 'sideways']  # 牛市/熊市/震荡
    )

    # 5. 风险分析
    risk_analysis = analyze_risk(
        results=training_results,
        metrics=['max_drawdown', 'volatility', 'concentration']
    )

    # 6. 生成反馈文本
    feedback = f"""
绩效分析:
- 最佳夏普比率: {best['sharpe']:.3f}
- 最差夏普比率: {worst['sharpe']:.3f}
- 改进空间: {(best['sharpe'] - worst['sharpe']):.3f}

特征重要性:
{format_feature_importance(feature_importance)}

成功模式:
{format_success_patterns(success_patterns)}

失败模式:
{format_failure_patterns(failure_patterns)}

市场环境:
{format_market_analysis(market_analysis)}

风险分析:
{format_risk_analysis(risk_analysis)}

改进建议:
{generate_improvement_suggestions(success_patterns, failure_patterns)}
"""

    return feedback

def analyze_feature_importance(results, methods):
    """
    特征重要性分析（金融版Lipschitz）

    方法 1: 相关性分析
    - 计算每个特征与收益的相关系数
    - 正相关 → 特征值大，收益也大
    - 负相关 → 特征值大，收益反而小

    方法 2: SHAP值
    - 训练一个简单模型（如线性回归）
    - 计算SHAP值
    - SHAP值绝对值 → 特征重要性

    方法 3: 信息增益
    - 计算特征与目标的信息增益
    - 信息增益大 → 预测能力强
    """
    importance = {}

    for method in methods:
        if method == 'correlation':
            # 相关性分析
            importance[method] = compute_correlation(results)
        elif method == 'shap':
            # SHAP值分析
            importance[method] = compute_shap(results)
        elif method == 'information_gain':
            # 信息增益
            importance[method] = compute_information_gain(results)

    return importance

def identify_success_patterns(best_result):
    """
    识别成功模式
    """
    patterns = []

    # 模式 1: 特征组合
    if 'momentum' in best_result['features'] and 'volatility' in best_result['features']:
        patterns.append("动量 + 波动率组合有效")

    # 模式 2: 时间尺度
    if has_multiple_timeframes(best_result['features']):
        patterns.append("多时间尺度特征捕捉趋势")

    # 模式 3: 风险管理
    if has_risk_management(best_result['reward_function']):
        patterns.append("内在奖励考虑风险调整")

    # 模式 4: 成交量确认
    if uses_volume_confirmation(best_result['features']):
        patterns.append("成交量确认价格趋势")

    return patterns

def identify_failure_patterns(worst_result):
    """
    识别失败模式
    """
    patterns = []

    # 模式 1: 过拟合
    if len(worst_result['features']) > 20:
        patterns.append("特征过多导致过拟合")

    # 模式 2: 缺少风险
    if no_risk_consideration(worst_result['reward_function']):
        patterns.append("内在奖励未考虑风险")

    # 模式 3: 单一信号
    if len(worst_result['features']) < 3:
        patterns.append("特征过少，信号单一")

    # 模式 4: 数值不稳定
    if has_numerical_issues(worst_result):
        patterns.append("数值稳定性问题")

    return patterns
```

## 五、FINSABER 提示词验证清单

### 5.1 验证清单实现

```python
# ========== FINSABER 提示词验证工具 ==========

class FINSABERPromptValidator:
    """
    FINSABER 提示词验证器
    """

    @staticmethod
    def validate_init_prompt(prompt_template):
        """
        验证初始化提示词
        """
        print("=== FINSABER 初始化提示词验证 ===\n")

        checks = []

        # 1. 金融语义丰富性
        print("【金融语义丰富性检查】")

        financial_terms = [
            ('收益率', 'return'),
            ('波动率', 'volatility'),
            ('动量', 'momentum'),
            ('趋势', 'trend'),
            ('技术指标', 'indicator')
        ]

        term_check = all(
            term[1] in prompt_template.lower()
            for term in financial_terms
        )
        checks.append(("金融术语完整性", term_check))

        indicator_check = any(
            indicator in prompt_template
            for indicator in ['SMA', 'EMA', 'RSI', 'MACD', 'ATR', 'OBV']
        )
        checks.append(("技术指标说明", indicator_check))

        risk_check = 'risk' in prompt_template.lower()
        checks.append(("风险意识", risk_check))

        # 2. 金融约束明确
        print("\n【金融约束明确性检查】")

        cost_check = 'commission' in prompt_template.lower() or 'cost' in prompt_template.lower()
        checks.append(("交易成本说明", cost_check))

        range_check = '[-100, 100]' in prompt_template or 'range' in prompt_template.lower()
        checks.append(("输出范围限制", range_check))

        boundary_check = ('divide' in prompt_template.lower() or
                         'zero' in prompt_template.lower() or
                         '1e-6' in prompt_template or
                         'epsilon' in prompt_template.lower())
        checks.append(("边界处理提示", boundary_check))

        # 3. 可执行验证
        print("\n【可执行验证检查】")

        function_check = 'def revise_state' in prompt_template
        checks.append(("函数签名", function_check))

        example_check = '```python' in prompt_template
        checks.append(("示例代码", example_check))

        numpy_check = 'numpy' in prompt_template.lower() or 'np.' in prompt_template
        checks.append(("NumPy使用", numpy_check))

        # 4. 迭代反馈
        print("\n【迭代反馈检查】")

        metric_check = any(
            metric in prompt_template.lower()
            for metric in ['sharpe', 'return', 'drawdown', 'win rate']
        )
        checks.append(("绩效指标说明", metric_check))

        # 5. 金融特有
        print("\n【金融特有检查】")

        timeseries_check = ('time' in prompt_template.lower() or
                           'sequence' in prompt_template.lower() or
                           'history' in prompt_template.lower())
        checks.append(("时序处理提示", timeseries_check))

        overfit_check = ('overfit' in prompt_template.lower() or
                        'generaliz' in prompt_template.lower())
        checks.append(("过拟合警告", overfit_check))

        nonstationary_check = ('non-stationary' in prompt_template.lower() or
                              'stationary' in prompt_template.lower())
        checks.append(("非平稳性提示", nonstationary_check))

        # 输出结果
        pass_count = sum(1 for _, result in checks if result)
        total_count = len(checks)

        for check, result in checks:
            status = "✅" if result else "❌"
            print(f"{status} {check}")

        print(f"\n通过率: {pass_count}/{total_count} ({pass_count/total_count*100:.1f}%)")

        if pass_count == total_count:
            print("🎉 所有检查通过！提示词设计完善。")
            grade = "A"
        elif pass_count >= total_count * 0.8:
            print("⚠️  大部分检查通过，建议优化缺失项。")
            grade = "B"
        elif pass_count >= total_count * 0.6:
            print("⚠️  部分检查未通过，需要改进。")
            grade = "C"
        else:
            print("❌ 多项检查未通过，提示词需要大幅改进。")
            grade = "D"

        return {
            'grade': grade,
            'pass_count': pass_count,
            'total_count': total_count,
            'pass_rate': pass_count / total_count,
            'checks': checks
        }

    @staticmethod
    def validate_cot_prompt(prompt_template):
        """
        验证COT反馈提示词
        """
        print("=== FINSABER COT反馈提示词验证 ===\n")

        checks = []

        # 1. 绩效分析
        checks.append(("包含夏普比率分析",
                     'sharpe' in prompt_template.lower()))

        checks.append(("包含回撤分析",
                     'drawdown' in prompt_template.lower()))

        # 2. 特征重要性
        checks.append(("包含相关性分析",
                     'correlation' in prompt_template.lower() or
                     'corr' in prompt_template.lower()))

        checks.append(("包含SHAP分析",
                     'shap' in prompt_template.lower()))

        # 3. 成功/失败模式
        checks.append(("包含成功模式识别",
                     'success' in prompt_template.lower() or
                     'best' in prompt_template.lower()))

        checks.append(("包含失败模式识别",
                     'fail' in prompt_template.lower() or
                     'worst' in prompt_template.lower()))

        # 4. 分析引导
        checks.append(("包含分析问题",
                     '(a)' in prompt_template or
                     '(b)' in prompt_template or
                     '(c)' in prompt_template))

        # 输出结果
        pass_count = sum(1 for _, result in checks if result)
        total_count = len(checks)

        for check, result in checks:
            status = "✅" if result else "❌"
            print(f"{status} {check}")

        print(f"\n通过率: {pass_count}/{total_count} ({pass_count/total_count*100:.1f}%)")

        return {
            'pass_count': pass_count,
            'total_count': total_count,
            'pass_rate': pass_count / total_count
        }
```

## 六、总结与实施建议

### 6.1 核心要点

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINSABER 提示词设计核心要点                                  │
└─────────────────────────────────────────────────────────────────────────────┘

1. 金融语义丰富性 ✅
   - 详细的金融术语解释
   - 技术指标计算示例
   - 市场规律说明

2. 金融约束明确 ✅
   - 交易成本考虑
   - 风险管理要求
   - 数值稳定性保障

3. 分类特征建议 ✅
   - 趋势指标
   - 动量指标
   - 波动率指标
   - 成交量指标
   - 微观结构指标

4. 具体计算示例 ✅
   - RSI、MACD、ATR等
   - 向量化操作
   - 边界处理

5. 多原则奖励设计 ✅
   - 趋势跟随
   - 风险调整
   - 极端值惩罚
   - 成交量确认

6. 金融特有反馈 ✅
   - 夏普比率分析
   - 相关性/SHAP分析
   - 过拟合检测
   - 市场环境分析
```

### 6.2 与LESR的对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  FINSABER vs LESR 提示词设计对比                               │
└─────────────────────────────────────────────────────────────────────────────┘

维度              │ LESR                │ FINSABER
─────────────────┼─────────────────────┼────────────────────────────────
状态描述          │ 简洁（物理量）      │ 详细（金融术语）
                  │ s[8]: velocity (m/s)│ s[0:19]: close prices ($)
                  │                     │ → 需要解释收益率、动量等
─────────────────┼─────────────────────┼────────────────────────────────
特征建议          │ 物理特征            │ 分类技术指标
                  │ 速度、能量、协调     │ 趋势、动量、波动率、成交量
─────────────────┼─────────────────────┼────────────────────────────────
示例代码          │ 简单计算            │ 复杂指标（RSI, MACD）
                  │ forward_vel = s[8]  │ rsi = 100 - 100/(1+rs)
─────────────────┼─────────────────────┼────────────────────────────────
奖励设计          │ 单一目标            │ 多目标权衡
                  │ reward = v - energy  │ 考虑趋势、风险、成交量
─────────────────┼─────────────────────┼────────────────────────────────
性能指标          │ 累积奖励            │ 夏普比率、回撤
                  │ Lipschitz常数        │ 相关系数、SHAP值
─────────────────┼─────────────────────┼────────────────────────────────
失败模式          │ 摔倒、震荡          │ 过拟合、过度交易
─────────────────┼─────────────────────┼────────────────────────────────
环境因素          │ 固定环境            │ 市场环境（牛熊震荡）
─────────────────┼─────────────────────┼────────────────────────────────
领域提示          │ 通常不需要          │ 强烈建议
                  │ (物理直觉足够)      │ (金融知识复杂)
```

### 6.3 实施建议

```python
# ========== 实施建议 ==========

"""
阶段1: MVP（2-3周）
├─ 使用基础版本的FINSABER_INIT_PROMPT
├─ 先实现5-8个核心特征
├─ 简单的内在奖励（趋势 + 风险）
└─ 验证端到端流程

阶段2: 优化（2-3周）
├─ 根据反馈优化Prompt
├─ 添加更多技术指标
├─ 改进内在奖励设计
└─ 实现COT反馈机制

阶段3: 完善（2-4周）
├─ 多轮迭代优化
├─ 市场环境识别
├─ 风险管理增强
└─ 过拟合控制

关键成功因素:
1. 详细的金融语义 → LLM理解数据
2. 具体的计算示例 → 减少错误
3. 分类特征建议 → 系统化思考
4. 多原则奖励设计 → 平衡目标
5. 金融特有反馈 → 持续改进
"""
```

---

**文档版本**: v1.0
**创建日期**: 2026-04-02
**作者**: LESR-FINSABER集成分析
**适用项目**: FINSABER量化交易框架
