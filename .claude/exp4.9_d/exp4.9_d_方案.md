# exp4.9_d 优化方案

> 基于 exp_4.9_b 实验结果的改进
> 讨论日期: 2026-04-09

## 一、exp_4.9_b 失败根因

### 问题 1：LLM 职责混乱
- LLM 同时负责生成技术指标（确定性数值计算）和设计 intrinsic_reward（需要智能判断）
- 技术指标是固定数学公式（SMA、RSI、MACD），让 LLM 写这些代码浪费 token 且容易出错
- Init 阶段验证失败率高（约 40-50% 样本不通过），大部分失败原因是指标计算代码有 bug

### 问题 2："只买不卖"
- C1（reward 与持仓绑定）+ B1（持仓感知）导致 DQN 学到"永远持仓"策略
- 大量 Trades=1（买入后不卖出），在牛市赢了但不是真正的策略能力
- 根因：持仓时 reward=价格变化（牛市永远正），卖出后 reward=0，交易还有手续费惩罚

### 问题 3：exp_4.9_b 做对了什么（应保留）
- NFLX 30%→60%：个股特异性 Prompt 有效
- W8 COVID 改善：部分机制在极端波动中有效
- W5 震荡市改善：手续费惩罚减少噪音交易

## 二、exp4.9_d 方案

### D1: 固定技术指标预计算（核心改动）

**思路**：技术指标是确定性数学计算，用固定代码预计算，不浪费 LLM 能力。

**具体做法**：

1. 新增 `feature_engine.py`，固定预计算 25-30 个技术指标：
   - 多时间窗口趋势（5/10/20 日 SMA, EMA, 价格相对 MA 偏离）
   - 动量指标（5/10/14 日 RSI, MACD 三线, 动量变化率）
   - 波动率指标（5/20 日历史波动率, ATR, 波动率比率）
   - 量价关系（OBV 趋势, 量价相关系数, 成交量比率）
   - 市场状态（波动率比率, 趋势强度 R², 价格位置, 量能异动）

2. `revise_state()` 变为固定函数，直接调用 `feature_engine.py`，**不再由 LLM 生成**

3. state_dim 固定为 120 + 25~30 = 145~150

**效果**：
- Init 阶段验证失败率从 ~40% 降到 0%（不需要验证指标计算代码）
- 节省 LLM token（指标代码约占 Prompt 输出的 60%）
- 所有实验使用相同的、经过验证的指标集，公平对比

### D2: LLM 职责聚焦 — 只设计 intrinsic_reward

**思路**：LLM 只负责需要"智能"的部分 — 判断什么状态该给正/负 reward。

**具体做法**：

1. LLM 只生成 `intrinsic_reward(enhanced_state)` 函数
2. Prompt 中明确告知 enhanced_state 的结构（120 维 OHLCV + 25~30 维预计算指标的完整语义）
3. LLM 不再需要写任何指标计算代码

**Prompt 结构变化**：
```
旧 Prompt: "请生成 revise_state 和 intrinsic_reward 两个函数"
新 Prompt: "以下是 enhanced_state 的完整结构（已预计算），请设计 intrinsic_reward 函数"

## Enhanced State Structure (已预计算)
- s[0:19]: 20 days closing prices
- s[20:39]: 20 days opening prices
...
- s[120:124]: 5/10/20日 SMA + 价格/SMA20 偏离
- s[124:130]: 5/10/14日 RSI + MACD线 + 信号线 + 柱状图
- s[130:134]: 5/20日历史波动率 + ATR + 波动率比率
- s[134:138]: OBV趋势 + 量价相关 + 成交量比率
- s[138:142]: 波动率比率 + 趋势强度 + 价格位置 + 量能异动

请基于以上特征，设计 intrinsic_reward 函数...
```

**效果**：
- LLM 只需关注"金融判断"，不需要写数学计算代码
- 生成的代码更短、更不容易出错
- 可以把省下的 token 用来生成更好的 reward 设计

### D3: 回退 C1 — reward 不与持仓绑定

**思路**：回退 exp_4.7 的 reward 方式，解决"只买不卖"问题。

**具体做法**：

1. **dqn_trainer.py**: 
   - 删除持仓追踪逻辑（C1）
   - reward 恢复为逐日价格变化：`reward = (next_price - current_price) / current_price`
   - 保留 B1（持仓感知：状态向量末尾仍有 position flag）
   - 保留 B2（手续费惩罚：每次交易扣 0.001）

2. 但保留 B1 的持仓标记：
   - DQN 仍然能看到 position flag，知道当前是否持仓
   - 只是不再把 reward 和持仓绑定
   - 这样 DQN 可以学到"持仓+价格跌 → 卖出"，而不是"永远持仓"

3. 训练循环逻辑：
   ```python
   # D3: reward 不与持仓绑定
   reward = (next_price - current_price) / current_price  # 始终计算价格变化
   
   if trade_executed:
       reward -= commission  # 保留手续费惩罚
   
   total_reward = reward + intrinsic_weight * intrinsic_r
   ```

**为什么这样能解决"只买不卖"**：
- reward 不再因为空仓就为 0
- DQN 需要同时学会"什么时候买"和"什么时候卖"
- 买入后价格跌 → 负 reward → DQN 学到应该卖出止损

### D4: 保留的改进（从 exp_4.9_b）

| 改动 | 保留？ | 说明 |
|------|--------|------|
| A1 个股特异性 Prompt | ✅ 保留 | NFLX 显著改善 |
| A2 市场状态检测 | ✅ 保留 | 但改为固定预计算，不由 LLM 生成 |
| A3 增加特征维度 | ✅ 保留 | 但改为固定预计算 |
| B1 持仓感知 | ✅ 保留 | position flag 仍加入状态向量 |
| B2 手续费惩罚 | ✅ 保留 | 每次交易扣 0.1% |
| C1 reward 与持仓绑定 | ❌ 回退 | 回退到 exp4.7 方式 |
| C2 intrinsic_weight 0.1 | ✅ 保留 | 继续使用 0.1 |

## 三、改动文件清单

| 文件 | 改动 |
|------|------|
| **新增** `feature_engine.py` | 固定预计算 25-30 个技术指标 |
| `dqn_trainer.py` | D3: 回退 C1（reward 不与持仓绑定）；保留 B1（position flag）；保留 B2（手续费） |
| `prompts.py` | D2: LLM 只生成 intrinsic_reward；告知完整 state 结构（含预计算指标语义） |
| `lesr_controller.py` | 适配：validate 只验证 intrinsic_reward；state_dim 固定 |
| `lesr_strategy.py` | 无变化（已有 B1 position flag） |
| `baseline.py` | 同步回退 C1 |
| config yaml | 无变化 |

## 四、预期效果

| 问题 | exp4.7 | exp_4.9_b | exp4.9_d 预期 |
|------|--------|-----------|--------------|
| NFLX 胜率 | 30% | 60% | ≥50%（保留个股 Prompt） |
| W8 COVID | 0/4 | 2/3 | ≥2/3（保留手续费+持仓感知） |
| "只买不卖" | 无此问题 | 严重 | 消除（回退 C1） |
| W1 温和牛市 | 4/4 | 0/3 | 恢复（reward 不绑定持仓） |
| Init 验证失败率 | ~40% | ~40% | **~0%**（指标固定预计算） |
| LLM token 使用 | 高 | 高 | **降低 ~50%**（只生成 reward） |
