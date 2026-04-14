# Exp4.9_c: 修正版 Regime-Aware LESR

## 设计哲学

```
正确的分工：
  确定性代码 → 算好 regime 数值，注入 state
  LLM        → 读取 regime 语义，决定特征/奖励策略
  框架层     → 维度保证、安全网
```

LLM 擅长语义信息处理，不擅长数值计算。
Regime 检测是数值计算 → 必须由确定性代码完成。
LLM 收到 regime 信息后，根据语义（"现在在危机中"→"应该保守"）来调整特征和奖励。

## Exp4.9 的三个实现 Bug（非方向错误）

### Bug 1: 维度一致性交给 LLM，35% 训练失败

**问题**: LLM 负责拼接 `[s, regime_vector, features]`，但 LLM 经常拼错 → 125/126/127/128/129/130 各种维度 → DQN 崩溃。

**修复**: 不让 LLM 拼接 regime_vector。框架层在 LLM 的 `revise_state` 输出之后，自动拼接：

```python
# 4.9 (错误): LLM 负责拼接
def revise_state(s, regime_vector):
    enhanced = np.concatenate([s, regime_vector, features])  # LLM 可能拼错
    return enhanced

# 4.9_c (正确): 框架层拼接
def revise_state(s):                    # 签名不变！
    features = compute_features(s)      # LLM 只管算特征
    return features                     # LLM 只返回新特征

# 框架层自动拼接:
enhanced_state = np.concatenate([s, regime_vector, llm_features])  # 确定性，不可能出错
```

**效果**: 消除 35% 的训练失败。LLM 代码更简单，只返回新增的特征向量。

### Bug 2: 5 维 regime_vector 语义模糊，LLM 难以利用

**问题**: `regime_vector = [0.5, 0.3, -0.2, 0.1, 0.0]` — LLM 看到这 5 个浮点数，不知道"0.5 是强趋势还是弱趋势"。

**修复**: 改为 3 维，每个维度有明确的语义标签，且归一化到容易理解的区间：

```python
regime_vector = [
    trend_direction,    # [-1, +1]: -1=下跌, 0=震荡, +1=上涨
    volatility_level,   # [0, 1]: 0=平静, 1=极端波动
    risk_level,         # [0, 1]: 0=安全, 1=高风险(近期大跌)
]
```

3 维比 5 维更容易被 LLM 理解和利用。同时 prompt 里给出清晰的阈值语义：

```
regime_vector[0] (trend_direction):
  > +0.3  → uptrend (consider momentum features)
  -0.3~+0.3 → sideways (consider mean-reversion features)
  < -0.3  → downtrend (be cautious)

regime_vector[1] (volatility_level):
  < 0.3  → calm market
  0.3~0.7 → normal
  > 0.7  → extreme volatility (reduce position signals)

regime_vector[2] (risk_level):
  < 0.3  → safe
  0.3~0.7 → elevated risk
  > 0.7  → dangerous (recent >10% drop, consider defensive)
```

### Bug 3: Safety net 阈值过于保守

**问题**: `crisis_signal > 0.7 → 强制 HOLD`，但 2020 V 型反弹被完全错过。

**修复**: 只保留 1 条最保守的规则，且阈值更高：

```python
# 4.9: 3 条规则
if crisis_signal > 0.7 and action == BUY: HOLD     # 太保守
if volatility > 0.85 and trend < -0.5: HOLD         # 不一定对
if consecutive_losses >= 3: HOLD                     # 剥夺学习机会

# 4.9_c: 1 条规则，阈值更高
if risk_level > 0.85 and action == BUY: reduce position scale to 30%
# 注意: 不是完全阻止，而是缩小仓位。且阈值从 0.7 提高到 0.85。
```

---

## 架构设计

```
Raw State (120d)
      │
      ├──────────────────────────────┐
      │                              │
      ▼                              ▼
regime_detector.py              revise_state(s)        ← LLM 生成的，签名不变
(确定性代码)                        │
  输出: regime_vector (3d)          ▼
      │                        llm_features (Nd)       ← LLM 只返回新增特征
      │                              │
      └──────────┬───────────────────┘
                 │
                 ▼  框架层自动拼接 (不可能出错)
        enhanced_state = [s(120) + regime(3) + features(N)]
                 │
                 ├──────────────────┐
                 ▼                  ▼
          intrinsic_reward()    regime_bonus()        ← 框架层确定性奖励
          (LLM 生成)            (简单规则)
                 │                  │
                 └────────┬─────────┘
                          ▼
                   total_intrinsic = intrinsic_reward + 0.5 * regime_bonus
                          │
                          ▼
                      DQN 训练
```

## 关键改动细节

### 1. regime_detector.py — 简化为 3 维

```python
def detect_regime(s: np.ndarray) -> np.ndarray:
    """
    输入: 120d interleaved state [close,open,high,low,vol,adj_close] x 20 days
    输出: 3d regime vector
      [0] trend_direction:  [-1, +1]
      [1] volatility_level: [0, 1]
      [2] risk_level:       [0, 1]
    """
    closes = np.array([s[i*6] for i in range(20)], dtype=float)
    highs = np.array([s[i*6+2] for i in range(20)], dtype=float)
    lows = np.array([s[i*6+3] for i in range(20)], dtype=float)
    volumes = np.array([s[i*6+4] for i in range(20)], dtype=float)

    # [0] trend_direction: MA(5) vs MA(20) relative distance
    ma5 = np.mean(closes[-5:])
    ma20 = np.mean(closes)
    trend = np.clip((ma5 - ma20) / (ma20 * 0.05 + 1e-8), -1, 1)

    # [1] volatility_level: recent ATR relative to historical
    ranges = (highs - lows) / (closes + 1e-8)
    recent_range = np.mean(ranges[-5:])
    hist_range = np.mean(ranges)
    hist_std = np.std(ranges) + 1e-10
    vol_z = (recent_range - hist_range) / hist_std
    volatility = np.clip((vol_z + 1) / 3, 0, 1)

    # [2] risk_level: max drawdown in recent 10 days
    window = closes[-min(10, len(closes)):]
    recent_high = np.max(window)
    current = closes[-1]
    dd = (recent_high - current) / (recent_high + 1e-8)
    risk = np.clip(dd / 0.15, 0, 1)  # 15% drawdown → risk=1.0

    return np.array([trend, volatility, risk], dtype=float)
```

### 2. prompts.py — 签名不变，prompt 增加语义指导

```python
def revise_state(s):
    """
    s: 120d raw state (interleaved OHLCV x 20 days)
       s[i*6 + 0] = close, s[i*6 + 1] = open,
       s[i*6 + 2] = high, s[i*6 + 3] = low,
       s[i*6 + 4] = volume, s[i*6 + 5] = adj_close

    Returns: new_features (1D numpy array)
    - Return ONLY the new features you compute
    - Do NOT include the original 120d state — the framework will prepend it
    - Do NOT include regime_vector — the framework will inject it
    - Return just your computed feature values
    """
    # Your feature computation here
    features = []
    ...
    return np.array(features)

def intrinsic_reward(enhanced_state):
    """
    enhanced_state layout:
    [0:120] = original raw state
    [120:123] = regime_vector:
      [120] trend_direction [-1,+1]: >+0.3 uptrend, <-0.3 downtrend
      [121] volatility_level [0,1]: >0.7 extreme volatility
      [122] risk_level [0,1]: >0.7 recent significant drop
    [123:] = LLM-generated features (your output from revise_state)

    CRITICAL: Use regime_vector to condition your reward:
    - trend > 0.3 + momentum aligned → positive reward
    - trend < -0.3 → cautious reward (reduce magnitude)
    - risk > 0.7 → negative reward for BUY signals
    - volatility > 0.7 → reduce all reward magnitudes
    """
    regime = enhanced_state[120:123]
    ...
    return reward
```

### 3. dqn_trainer.py — 框架层拼接 + regime_bonus

```python
# 在 _get_cached_state 中
def _get_cached_state(self, data_loader, date):
    raw_state, regime_vector = self.extract_state(data_loader, date)
    
    # LLM 只返回新特征（不含 raw state 和 regime）
    llm_features = self.revise_state(raw_state)
    
    # 框架层确定性拼接 — 维度永远正确
    enhanced = np.concatenate([raw_state, regime_vector, llm_features])
    return enhanced

# regime_bonus 在训练循环中
def compute_regime_bonus(self, raw_state, action):
    """只有 1 条规则：极端风险时不鼓励买入"""
    regime = detect_regime(raw_state)
    risk_level = regime[2]
    
    if risk_level > 0.85 and action == 0:  # BUY
        return -3.0  # 温和惩罚，不是完全阻止
    return 0.0
```

### 4. _validate_code — 简化验证

```python
# 4.9: 需要验证 2 参数签名 + regime_vector 在正确位置
test_regime = np.zeros(5)
enhanced = module.revise_state(test_state, test_regime)
assert np.allclose(enhanced[120:125], test_regime)  # 经常失败

# 4.9_c: 只验证输出是 1D numpy 数组
features = module.revise_state(test_state)
assert isinstance(features, np.ndarray)
assert features.ndim == 1
# 维度完全由框架层决定，不可能不一致
```

---

## 文件改动清单

| 文件 | 改动 | 说明 |
|------|------|------|
| `regime_detector.py` | 重写 | 5维→3维，修正 state 解析 |
| `prompts.py` | 重写 | 签名不变，prompt 加语义指导 |
| `dqn_trainer.py` | 改 3 处 | 框架层拼接 + regime_bonus + 简化验证 |
| `lesr_controller.py` | 改 2 处 | 验证逻辑简化 + state_dim 计算 |
| `lesr_strategy.py` | 改 1 处 | 框架层拼接 |
| `feature_analyzer.py` | 不改 | |
| `baseline.py` | 不改 | |

---

## 与 4.7 / 4.9 的对比

| 维度 | 4.7 | 4.9 | 4.9_c |
|------|-----|-----|-------|
| revise_state 签名 | `(s)` | `(s, rv)` ❌ | `(s)` ✓ |
| regime 信息 | 无 | 5维外部注入 | 3维外部注入 |
| 谁拼接 state | LLM | LLM ❌ | **框架层** ✓ |
| intrinsic_reward | LLM 全权 | LLM 读 regime 分支 | LLM 读 regime + 框架层 regime_bonus |
| safety net | 无 | 3条规则，太激进 | 1条规则，保守 |
| 训练成功率 | ~90% | 65% ❌ | **预期 ~90%** ✓ |
| LLM 代码复杂度 | 低 | 高 | **低** ✓ |

## 预期效果

- 训练成功率：65% → ~90%（消除维度不一致）
- regime 信息有效利用：LLM 能读懂 3 维语义明确的 regime_vector
- safety net 不过度干预：只在高风险(risk>0.85)时温和降仓
- 整体胜率：预期 55% → 65%+

---

## 补充：危机期策略优化

### 问题分析

Exp4.9 的 crisis Sharpe = -5.24，88% 的 crisis 期组都亏损。有两个具体原因：

**原因 1: Regime 检测有滞后**

```
实际时间线:
  Day 1-3:  正常上涨 → regime=sideways → DQN 买入
  Day 4-7:  开始下跌 → regime=trend_down → DQN 还在持仓
  Day 8-10: 暴跌加速 → regime=crisis → safety net 触发，但已经亏了 15%
```

等 regime 检测到 crisis 时，仓位已经深度被套。safety net 说"不许再买"，但没解决"已持有的怎么办"。

**原因 2: Prompt 只说"危机时给负奖励"，太笼统**

LLM 不知道具体应该做什么：
- 危机中应该止损卖出？还是持有等待反弹？
- 不同类型的危机（慢跌 vs 急跌）策略应该不同
- 当前 prompt 没有区分

### 解决方案：两层危机防御

#### 第一层：intrinsic_reward 中的危机奖励分层（给 LLM 更细的语义）

prompt 中不再是笼统的"crisis > 0.5 → 负奖励"，而是给出具体的决策逻辑：

```
## Regime-Conditioned Reward Logic (CRITICAL)

enhanced_state[120:123] contains regime_vector:
  [120] trend_direction: [-1, +1]
  [121] volatility_level: [0, 1]  
  [122] risk_level: [0, 1]

Your intrinsic_reward MUST implement this priority logic:

Priority 1 — RISK MANAGEMENT (check FIRST):
  if risk_level > 0.6:
    - If enhanced_state suggests an OPEN position (e.g., recent buy signals in features):
      → Return NEGATIVE reward (encourage risk awareness)
    - If enhanced_state suggests NO position:
      → Return small NEGATIVE reward for any BUY signal (discourage new entries)
  
  if risk_level > 0.8:
    → Return STRONG NEGATIVE reward (-30 to -50) for any BUY-aligned features
    → Return MILD POSITIVE reward (+5 to +10) for SELL-aligned features (encourage exit)

Priority 2 — TREND FOLLOWING:
  if |trend_direction| > 0.3 and volatility < 0.5:
    → Reward momentum-aligned features
    → trend > 0.3 + upward features → positive
    → trend < -0.3 + downward features → positive (correct bet)

Priority 3 — SIDEWAYS / NORMAL:
  if |trend_direction| < 0.3:
    → Reward mean-reversion features
    → Penalize breakout-chasing features

Priority 4 — HIGH VOLATILITY (but not crisis):
  if volatility > 0.6 and risk_level < 0.4:
    → Reduce reward magnitude by 50% (uncertain market)
```

**关键改进**：不是"危机=不买"，而是分优先级的决策链：
- risk > 0.8 → 鼓励卖出（正奖励），强烈阻止买入（负奖励）
- risk 0.6-0.8 → 温和警示（小负奖励）
- 这样 DQN 学到的不只是"危机时不动"，而是"危机时应该止损"

#### 第二层：框架层的 trailing stop（不通过 LLM/DQN，确定性规则）

在 `dqn_trainer.py` 的训练和评估中，加一个简单的**动态止损**：

```python
def compute_regime_bonus(self, raw_state, regime_vector, action, portfolio_state):
    """
    框架层奖励，补充 LLM 的 intrinsic_reward。
    只做一件事：在风险升高时鼓励止损。
    """
    risk_level = regime_vector[2]
    
    # 规则 1: 高风险时鼓励卖出（止损）
    if risk_level > 0.6 and action == 1:  # SELL
        return +5.0  # 正奖励 → DQN 学到"高风险时卖出是对的"
    
    # 规则 2: 极高风险时强烈阻止买入
    if risk_level > 0.85 and action == 0:  # BUY
        return -5.0  # 负奖励 → DQN 学到"极端风险不入场"
    
    return 0.0
```

**与 exp4.9 safety net 的区别**：

| | Exp4.9 | Exp4.9_c |
|---|---|---|
| 高风险+BUY | 强制 HOLD（硬拦截） | 负奖励 -5（软引导） |
| 高风险+SELL | 无特殊处理 | 正奖励 +5（鼓励止损） |
| 效果 | DQN 没学到东西 | DQN 学到"风险高时应该卖" |

硬拦截的问题：DQN 永远不会在 crisis 中做决策，所以**永远不会学到**怎么应对危机。
软引导的好处：DQN 在 crisis 中做了决策（卖出），得到了正奖励，**学会了危机应对**。

#### 完整的奖励公式

```python
total_reward = extrinsic_reward                                          # 实际盈亏
             + intrinsic_weight * intrinsic_reward(enhanced_state)       # LLM 的语义奖励
             + regime_bonus_weight * regime_bonus(raw_state, action)     # 框架层的风险奖励
```

其中 `regime_bonus_weight = 0.01`（比 intrinsic_weight 小，只做辅助引导）。

---

### 预期效果

| 场景 | Exp4.9 问题 | Exp4.9_c 预期 |
|------|------------|---------------|
| trend_up | Sharpe 3.36 ✓ | 保持 3.36 |
| crisis + 已持仓 | Sharpe -5.24，DQN 不知道止损 | DQN 学到止损 → Sharpe 改善到 -1~-2 |
| crisis + 未持仓 | safety net 硬阻止 | regime_bonus 软引导 → DQN 自己学会不买 |
| V型反弹 | 完全错过 | 不硬拦截，DQN 可以在低点买入 |
| overall crisis Sharpe | -5.24 | **预期 -1 到 -2** |
---

## Exp4.9_c 实验结果分析 (10窗口完成)

### 整体结果

| 版本 | LESR 胜率 (配对) |
|------|-----------------|
| 4.7 | 44% (15/34) |
| 4.9 | 55% (16/29) |
| **4.9_c** | **65% (22/34)** |

净改善: +7 组 (16 翻转赢 vs 9 翻转输)

### 9 组失败的根因

#### 原因 1: Baseline DQN 随机性 — 5/9 组，非 4.9_c 的问题

这些组不是 LESR 退化了，而是 Baseline 的 DQN 训练结果不同：

| 窗口/股票 | 4.7 Base | 4.9_c Base | 变化 |
|----------|---------|-----------|------|
| W1/TSLA | -0.93 | +0.33 | Base 从亏变盈 |
| W2/MSFT | -1.54 | +1.21 | Base 从亏变盈 |
| W3/NFLX | -0.46 | +1.10 | Base 从亏变盈 |
| W7/AMZN | -0.39 | +0.45 | Base 从亏变盈 |
| W10/NFLX | -1.67 | +1.34 | Base 从亏变盈 |

**本质**: DQN 训练随机性导致 Baseline 结果波动很大（Sharpe 差距 0.84~3.01），这不是实验设计能控制的。

**解决方案**: 
- 在同一次实验中共享 Baseline 训练结果（用固定随机种子）
- 或跑多次取平均

#### 原因 2: LESR 策略真正退化 — 3/9 组

| 窗口/股票 | 4.7 LESR | 4.9_c LESR | 下降 | 可能原因 |
|----------|---------|-----------|------|---------|
| W2/AMZN | 1.47 | -0.09 | 1.56 | 2018震荡市，regime可能误判trend_direction |
| W7/MSFT | 1.30 | 0.66 | 0.64 | 2016温和牛，intrinsic_reward 可能过于保守 |
| W9/MSFT | 2.36 | 1.19 | 1.17 | 训练集含金融危机，策略可能过度保守 |

**共同特征**: 
- 都是训练集和测试集市场机制差异大的窗口
- W2 训练 2013-2015 → 测试 2018 震荡
- W7 训练 2011-2013 → 测试 2016 温和牛
- W9 训练 2008-2010(含金融危机) → 测试 2013 复苏

**可能原因**: 
- regime_detector 在震荡/温和市中对 risk_level 估算偏高，intrinsic_reward 给了过多 SELL 正奖励
- 训练集含危机数据时，regime_bonus 的 "risk > 0.6 → 鼓励 SELL" 规则可能让 DQN 过度学习 "总是卖出"
- 3 维 regime 信息虽然比 5 维简单，但 regime_bonus 权重 0.01 可能对某些股票偏高

#### 原因 3: 边际差异 — 1/9 组
- W3/TSLA: LESR 1.97→1.87 vs Base 1.84→1.99，两边都很好，差距微小

### 5 组缺失结果

| 窗口/股票 | 原因 |
|----------|------|
| W1/AMZN | LLM 代码 off-by-one bug (high[1:]-close[1:-1] 形状不匹配) |
| W1/MSFT | 同上 |
| W5/TSLA | 早期数据无此 ticker（同 4.7） |
| W9/TSLA | 同上 |
| W10/TSLA | 同上 |

W1 的 2 组缺失是因为 _validate_code 不够严格——只测了 zeros 和 ones，没测更真实的输入（如 linspace 上升趋势）。

---

## 下一步改进方向

### 优先级 1: 修复 Validation（已修复）

_validate_code 改为测试 4 种输入（zeros, ones, random, linspace），catch off-by-one bugs。

### 优先级 2: 调优 regime_bonus 权重和阈值

当前 regime_bonus_weight=0.01，risk > 0.6 就鼓励 SELL。但 3 组真正退化的案例都是"温和市"或"复苏市"，可能 risk_level 被高估了。

**改进**:
- 将 SELL 正奖励的阈值从 risk > 0.6 提高到 risk > 0.7
- 或降低 regime_bonus_weight 从 0.01 到 0.005

### 优先级 3: Baseline 随机性控制

5/9 的"失败"是 Baseline 运气导致的。解决方案：
- 给 Baseline DQN 设置固定随机种子
- 或在 _eval_worker 中对 Baseline 跑 3 次取平均 Sharpe

### 优先级 4: 缺失结果修复

W1/AMZN 和 W1/MSFT 的缺失是 LLM 代码 bug。修复 validation 后会自动拒绝这些 buggy code。

### 预期改进

如果修复以上问题：
- 缺失的 2 组（W1/AMZN, W1/MSFT）可能有结果
- 3 组真正退化可能改善 1-2 组
- 总胜率可能从 65% 提升到 70%+
