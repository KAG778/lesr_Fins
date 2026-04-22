# exp_4.9_b 设计方案：LESR 机制优化

> 基于 exp4.7 实验结果的失败分析，针对核心问题进行优化
> 讨论日期：2026-04-09

---

## 一、exp4.7 失败原因回顾

| 失败模式 | 典型案例 | 根因 |
|----------|---------|------|
| 特征失效 | W8 (2020) 0/4 全败 | LLM 特征基于历史模式，极端波动时完全失效 |
| 不交易 | W5 NFLX 0笔 | DQN 学会"不确定就不动"，保守过度 |
| 过度交易 | W5 AMZN 21笔 vs Base 3笔 | LLM 特征产生噪音信号，频繁进出 |
| 方向错误 | W9 NFLX -16.30pp | 特征在强趋势中给出反向信号 |
| 个股适应性差 | NFLX 3胜/7负 = 30% | 统一特征框架无法适应不同股票 |

**根因排序**：特征失效(主因) > 交易行为不稳健(次因) > 个股适应性差

---

## 二、确认的优化方案（共 7 项）

### A1: 个股特异性 Prompt

**问题**：当前所有股票用完全相同的 Prompt，LLM 不知道为哪只股票生成特征。

**改动位置**：`prompts.py` 的 INITIAL_PROMPT 和 get_iteration_prompt

**具体内容**：在 Prompt 末尾追加股票信息，全部从训练期量价数据计算：

```
## Target Stock Information
- Ticker: {ticker}
- Training period daily volatility: {vol:.2f}%
- Training period total return: {total_return:.2f}%
- Training period avg daily volume: {avg_volume:.0f}

Optimize features specifically for this stock's risk and return profile.
```

**数据来源**：
- 日均波动率：`np.std(returns) * 100`，来自训练期收盘价
- 总涨跌幅：`(期末收盘价 - 期初收盘价) / 期初收盘价 * 100`
- 日均成交量：训练期 volume 的均值

**无泄露风险**：所有统计量只用训练期数据。

---

### A2: 市场状态检测（Regime Detection）

**问题**：2020 COVID 暴跌 + V 型反转等极端波动中，技术特征全部失效。

**改动位置**：`prompts.py` + `dqn_trainer.py`

**具体内容**：在 `revise_state` 中要求 LLM 新增 `detect_regime()` 逻辑，输出 4 个 regime 特征：

```python
# 全部从 20 日量价数据计算
regime_features = [
    volatility_ratio,    # 当前5日波动率 / 20日波动率 (>2 = 极端波动)
    trend_strength,      # 线性回归 R² (接近1 = 强趋势)
    price_position,      # 当前价 / 20日最高-最低范围 [0,1]
    volume_ratio,        # 当前5日均量 / 20日均量 (>2 = 量能异动)
]
```

**效果**：DQN 能感知当前市场状态，在极端波动时自动降低交易频率。

---

### A3: 增加新特征维度到 20-30 个

**问题**：当前只生成 5-7 个新特征，占 120 维原始状态的不到 5%，DQN 容易忽略。

**改动位置**：`prompts.py` 的 INITIAL_PROMPT

**具体内容**：修改 Prompt 要求 LLM 生成多类别、多时间窗口的特征：

```
特征类别要求（共 20-30 个新特征）：

1. 多时间窗口趋势指标（~8个）
   - 5日/10日/20日 SMA
   - 5日/10日/20日 EMA
   - 短期均线与长期均线差值

2. 动量指标（~6个）
   - 5日/10日/14日 RSI
   - MACD (线值 + 信号线 + 柱状图)
   - 动量变化率

3. 波动率指标（~4个）
   - 历史波动率 (5日/20日)
   - ATR (14日)
   - 波动率变化率

4. 量价关系指标（~4个）
   - OBV 趋势
   - 量价相关系数
   - 成交量比率

5. 市场状态指标（~4个，即 A2 的 regime 特征）
   - 波动率比率
   - 趋势强度
   - 价格位置
   - 量能异动
```

**效果**：新特征占比提升到 15-20%（140-150维总状态），DQN 无法忽略。

---

### B1: 持仓感知

**问题**：当前 intrinsic_reward 只接收 state，不知道是否持仓，导致"不交易"和"过度交易"。

**改动位置**：`dqn_trainer.py` 的 `train()` 和 `evaluate()`

**具体内容**：

1. 在状态向量末尾追加 1 位持仓标记：

```python
# 在 dqn_trainer.py 的 train() 和 evaluate() 中
position_flag = np.array([1.0 if current_position == 1 else 0.0])
enhanced_state = np.concatenate([enhanced_state, position_flag])
```

2. 修改 Prompt 告知 LLM：

```
## State Vector Structure
- s[0:119]: 120 dims of raw OHLCV data
- s[120:???]: LLM-generated features
- s[-1]: Position flag (1.0 = holding position, 0.0 = no position)

The intrinsic_reward function should use the position flag to:
- Encourage buying when trend is clear and position = 0
- Encourage selling when trend weakens and position = 1
- Penalize excessive trading
```

3. 相应更新 `state_dim`：`state_dim = revise_state_dim + 1`

**效果**：intrinsic_reward 能区分"该买"和"该卖"，减少无效交易。

---

### B2: 交易频率惩罚（手续费显式化）

**问题**：Prompt 中写明手续费 0.1%，但训练 reward 中没有体现，DQN 不知道频繁交易有成本。

**改动位置**：`dqn_trainer.py` 的 `train()` 方法

**具体内容**：在 reward 计算中加入交易成本惩罚：

```python
position = 0  # 追踪持仓
prev_action = 2  # 初始为 Hold

for i, date in enumerate(dates):
    action = dqn.select_action(state, epsilon)
    
    # 判断是否发生交易
    trade_executed = False
    if action == 0 and position == 0:   # Buy 且空仓 → 执行买入
        position = 1
        trade_executed = True
    elif action == 1 and position == 1: # Sell 且持仓 → 执行卖出
        position = 0
        trade_executed = True
    
    # 计算 reward（与持仓绑定）
    if position == 1:
        reward = (next_price - current_price) / current_price
    else:
        reward = 0.0
    
    # 交易手续费惩罚
    if trade_executed:
        reward -= 0.001  # 0.1% commission
```

**效果**：DQN 学到频繁交易有成本，自然倾向于减少无效交易。

---

### C1: 训练-评估一致性改进

**问题**：当前训练 reward 是逐日价格变化（与持仓无关），但评估收益与持仓绑定，目标不一致。

**改动位置**：`dqn_trainer.py` 的 `train()` 方法（第 199-276 行）

**具体内容**：将 B2 中的改动整合，使训练循环追踪持仓状态：

```python
def train(self, train_data_loader, start_date, end_date, max_episodes=100):
    dates = [d for d in ...]
    
    # 预计算特征（保持不变）
    for date in dates:
        self._get_cached_state(train_data_loader, date)
    
    for episode in range(max_episodes):
        position = 0  # 当前持仓状态
        episode_reward = 0
        
        for i, date in enumerate(dates):
            enhanced_state = self._get_cached_state(train_data_loader, date)
            if enhanced_state is None:
                continue
            
            # 追加持仓标记（B1）
            pos_flag = np.array([1.0 if position == 1 else 0.0])
            state_with_pos = np.concatenate([enhanced_state, pos_flag])
            
            action = self.dqn.select_action(state_with_pos, epsilon)
            
            current_price = train_data_loader.get_ticker_price_by_date(self.ticker, date)
            
            # 执行动作 + 更新持仓（B2）
            trade_executed = False
            if action == 0 and position == 0:
                position = 1
                trade_executed = True
            elif action == 1 and position == 1:
                position = 0
                trade_executed = True
            
            # 计算 reward（与持仓绑定，C1）
            if i < len(dates) - 1:
                next_date = dates[i + 1]
                next_price = train_data_loader.get_ticker_price_by_date(self.ticker, next_date)
                
                if position == 1:
                    reward = (next_price - current_price) / current_price
                else:
                    reward = 0.0
                
                # 手续费惩罚（B2）
                if trade_executed:
                    reward -= 0.001
                
                # Intrinsic reward
                intrinsic_r = self.intrinsic_reward(enhanced_state)
                total_reward = reward + self.intrinsic_weight * intrinsic_r
                
                # 下一状态也带持仓标记
                next_enhanced = self._get_cached_state(train_data_loader, next_date)
                if next_enhanced is not None:
                    next_pos_flag = np.array([1.0 if position == 1 else 0.0])
                    next_state_with_pos = np.concatenate([next_enhanced, next_pos_flag])
                else:
                    next_state_with_pos = state_with_pos
            else:
                next_state_with_pos = state_with_pos
                total_reward = 0
            
            # state_dim 需要包含持仓位
            self.buffer.push(state_with_pos, action, total_reward, next_state_with_pos, False)
            
            episode_reward += total_reward
        
        self._soft_update_target()
        print(f"Episode {episode}/{max_episodes}, Epsilon: {epsilon:.3f}")
    
    return self._get_training_summary()
```

**效果**：训练目标与评估目标完全一致，DQN 学到的是真正的交易策略。

**注意**：此改动将 B1 和 B2 整合在同一处代码中，实际实现时合并处理。

---

### C2: 提高 intrinsic reward 权重

**问题**：`intrinsic_weight = 0.02` 太小，intrinsic reward 几乎不影响训练。

**改动位置**：`config yaml` + `dqn_trainer.py` 的默认值

**具体内容**：

```yaml
# config 中新增
intrinsic_weight: 0.1  # 从 0.02 提高到 0.1
```

```python
# dqn_trainer.py 构造函数默认值修改
def __init__(self, ..., intrinsic_weight: float = 0.1):  # 原来是 0.02
```

**效果**：intrinsic reward 有实际影响力，但不会主导外部 reward。

---

## 三、暂缓方案

| 编号 | 方案 | 原因 |
|------|------|------|
| C3 | LESR + Base 集成（Sharpe 加权投票） | 先看 A1-C2 的效果，如仍不理想再加 |

## 四、不加的内容

| 内容 | 原因 |
|------|------|
| 外部基本面数据（EPS、营收等） | 数据泄露风险高，且需要额外数据管道 |
| 行业标签 | 不用量价以外的数据 |

## 五、改动文件清单

| 文件 | 改动内容 |
|------|---------|
| `prompts.py` | A1: 加入个股信息模板；A2: 加入 regime 特征要求；A3: 增加特征数量要求；B1: 加入持仓标记说明 |
| `dqn_trainer.py` | B1: 状态向量追加持仓位；B2: 交易手续费惩罚；C1: 训练循环持仓追踪；C2: intrinsic_weight 默认值 |
| `config.yaml` | A1: 加入个股信息字段；C2: intrinsic_weight 配置 |

## 六、实现优先级

建议实现顺序（考虑依赖关系）：

1. **C1 + B1 + B2**（一起改 dqn_trainer.py 的 train 方法）
2. **C2**（改 config 默认值）
3. **A1 + A2 + A3**（一起改 prompts.py）

这样先改训练机制，再改 Prompt，结构清晰。

## 七、数据泄露防护

所有新增信息来源确认：

| 数据 | 来源 | 泄露风险 |
|------|------|---------|
| 训练期波动率 | 训练期收盘价 | 无 |
| 训练期涨跌幅 | 训练期收盘价 | 无 |
| 训练期均量 | 训练期成交量 | 无 |
| Regime 特征 | 20 日量价滑动窗口 | 无 |
| 持仓标记 | 运行时状态 | 无 |
| 交易手续费 | 固定 0.1% | 无 |

**原则**：所有特征和信息只用当前时间点及之前的数据，不使用任何未来信息。
