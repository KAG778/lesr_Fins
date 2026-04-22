# Exp4.9: 机制感知的 LESR 优化方案

## 目标

**只优化 LESR pipeline**（LLM prompt → revise_state/intrinsic_reward → feature analysis → COT 反馈），
**不动 DQN 架构、不动 baseline**。

解决的核心问题：训练/验证/测试跨市场机制时，LLM 生成的特征和奖励函数无法泛化。

## 问题本质

当前 4.7 中 LLM 生成的是**一套无差别策略**：
- `revise_state` 对所有市场状态计算相同的特征
- `intrinsic_reward` 用统一的阈值逻辑

但最优策略因机制而异：
- 牛市 → 趋势跟踪，鼓励追涨
- 震荡 → 均值回归，低买高卖
- 崩盘 → 防守，抑制入场

当前 pipeline 在温和牛市胜率 100%，极端波动胜率 50%，根本原因是 LLM 没有"机制"的概念。

## 解法：给 LESR 注入机制感知

### 改动范围（仅 LESR 层）

```
                    ┌──────────────────────────────┐
                    │   Raw State (120d)            │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │  regime_detector.py (新增)    │  ← 确定性规则模块
                    │  计算 regime_vector (5d)      │     非 LLM 生成，非学习型
                    └──────────┬───────────────────┘
                               │
               ┌───────────────▼───────────────┐
               │  prompts.py (改写)             │  ← 核心改动
               │  Prompt 要求 LLM 的函数接收     │
               │  regime_vector 作为输入          │
               │  不同机制分支不同的特征/奖励      │
               └───────────────┬───────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │  LLM 生成的代码 (改)          │
                    │  revise_state(s, regime_vec)  │  ← 新签名
                    │  intrinsic_reward(enhanced_s) │     内含 regime 维度
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │  dqn_trainer.py (微调)        │  ← 只改 extract_state
                    │  自动计算 regime_vector       │     和 revise_state 调用方式
                    │  注入到 state 中              │     DQN 网络结构不变
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │  feature_analyzer.py (增强)   │  ← 按机制分组分析
                    │  COT 按机制报告特征效果       │     让迭代反馈更精准
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │  DQN (不动)                   │
                    │  3 层 MLP, action=3           │  ← 完全不变
                    └──────────────────────────────┘
```

### 模块详细设计

---

#### 1. `regime_detector.py` (新增)

纯数学规则，无训练，无 LLM。从 120d raw state 计算 5 维 regime vector。

```python
def detect_regime(s: np.ndarray) -> np.ndarray:
    """
    输入: s (120d raw state)
    输出: regime_vector (5d float)
    
    s[0:20]  = close prices
    s[20:40] = open prices
    s[40:60] = high prices  
    s[60:80] = low prices
    s[80:100] = volume
    
    返回:
    [0] trend_strength:     [-1, +1] 趋势方向和强度
    [1] volatility_regime:  [0, 1]   波动率水平 (相对历史)
    [2] momentum_signal:    [-1, +1] 动量方向
    [3] meanrev_signal:     [-1, +1] 均值回归机会
    [4] crisis_signal:      [0, 1]   危机警报级别
    """
```

**计算逻辑：**

| 维度 | 算法 | 含义 |
|------|------|------|
| trend_strength | (MA(5) - MA(20)) / MA(20)，clip 到 [-1, 1] | +1=强上涨趋势, -1=强下跌 |
| volatility_regime | (ATR(14)/price - mean) / std，clip 到 [0, 1] | 0=低波动, 1=极端高波动 |
| momentum_signal | z-score of recent 5-day ROC | 正=上涨动量, 负=下跌动量 |
| meanrev_signal | Bollinger %B - 0.5，clip 到 [-1, 1] | 正=价格偏高(回落概率), 负=偏低 |
| crisis_signal | max(极端回撤分, 放量下跌分) | 0=正常, 1=危机中 |

**为什么不用 LLM 生成这个：**
- 机制检测是"地面实况"，需要 100% 确定性
- 如果 LLM 生成的检测器本身就有错，后面所有逻辑都建立在错误基础上
- 这 5 个指标是标准的、经过验证的，没有"优化空间"

---

#### 2. `prompts.py` (重写)

**核心改动：LLM 生成的函数必须感知 regime。**

##### INITIAL_PROMPT 变化

函数签名改变：
```python
# 4.7 (旧)
def revise_state(s):
    ...

# 4.9 (新)  
def revise_state(s, regime_vector):
    ...
```

Prompt 中新增的指导：

```
## Market Regime Information (NEW)

Your revise_state function now receives a regime_vector (5-dimensional):
- regime_vector[0]: trend_strength [-1, 1]
- regime_vector[1]: volatility_regime [0, 1]
- regime_vector[2]: momentum_signal [-1, 1]
- regime_vector[3]: meanrev_signal [-1, 1]
- regime_vector[4]: crisis_signal [0, 1]

## Key Requirement: Regime-Conditioned Features

Your features should be MEANINGFUL under the current regime:
- When trend is strong (|trend_strength| > 0.3): compute trend-following features
- When market is sideways (|trend_strength| < 0.15): compute mean-reversion features
- When crisis_signal > 0.5: compute defensive features (drawdown rate, max loss)

## Key Requirement: Regime-Conditioned Reward

Your intrinsic_reward MUST branch on regime:
- STRONG TREND + momentum aligned → positive reward (趋势行情, 追涨杀跌是对的)
- SIDEWAYS + meanrev opportunity → mild positive reward (震荡行情, 低买高卖)  
- HIGH VOLATILITY + action conflicts with trend → negative reward (高波动, 减少交易)
- CRISIS (signal > 0.5) → strong negative reward for any entry (危机中, 不入场)
```

##### COT Prompt 变化

反馈按机制分组：

```
Performance by Regime:
  TREND_UP periods (47 days): Sharpe=1.2, 12 trades
  SIDEWAYS periods (32 days): Sharpe=0.3, 8 trades  
  HIGH_VOL periods (15 days): Sharpe=-0.8, 6 trades  ← 问题在这里！
  CRISIS periods (5 days): Sharpe=-2.1, 3 trades     ← 灾难！

Analysis: Your RSI-based features work well in trend markets but fail in 
high volatility. Suggestion: add ATR-based scaling to reduce position 
signals when volatility_regime > 0.7.
```

这比 4.7 的全局反馈精准得多——LLM 知道"我的策略在哪种市场不行"。

##### Iteration Prompt 变化

历史经验中也包含机制分组的分析结果，让 LLM 能针对性地改进。

---

#### 3. `lesr_controller.py` (增强)

改动点：
1. `_validate_code()` 验证新签名 `revise_state(s, regime_vector)`
2. `_train_ticker_worker` 中计算 regime_vector 并传入
3. 其余优化循环逻辑不变

验证测试：
```python
test_state = np.zeros(120)
test_regime = np.zeros(5)
enhanced = module.revise_state(test_state, test_regime)  # 新签名
intrinsic_r = module.intrinsic_reward(enhanced)
assert enhanced.shape[0] >= 125  # 120 + 5 regime + extra
```

---

#### 4. `dqn_trainer.py` (最小改动)

**只改两处，DQN 网络结构完全不变：**

1. `extract_state` 末尾调用 `detect_regime` 拼接到 state 里
2. 调用 `revise_state(s, regime_vector)` 而非 `revise_state(s)`

```python
# 旧
enhanced_state = self.revise_state(raw_state)

# 新
regime_vector = detect_regime(raw_state)
enhanced_state = self.revise_state(raw_state, regime_vector)
```

DQN 看到的 state_dim 变大了（120 + 5 + N），网络结构用同一套 MLP 自动适配。

---

#### 5. `feature_analyzer.py` (增强)

新增 `analyze_features_by_regime()` 函数：

```python
def analyze_features_by_regime(states, rewards, regime_labels, original_dim):
    """
    按市场机制分组计算特征重要性
    
    输入:
    - states: 所有 enhanced states
    - rewards: 对应奖励
    - regime_labels: 每条数据的主导机制标签
    - original_dim: 原始维度 (120)
    
    输出:
    - global_importance: 全局特征重要性
    - regime_importance: {
        'trend_up': importance_array,
        'trend_down': ...,
        'sideways': ...,
        'high_vol': ...,
        'crisis': ...
      }
    """
```

COT 反馈利用这个信息告诉 LLM："特征 X 在趋势市重要度 0.8，在震荡市 0.2"。

---

#### 6. `lesr_strategy.py` (微调)

回测时也注入 regime_vector：
```python
regime_vector = detect_regime(raw_state)
enhanced_state = self.revise_state(raw_state, regime_vector)
```

加一个简单的**后处理安全网**（可选，2 行代码）：
```python
if enhanced_s[124] > 0.7 and action == 0:  # crisis_signal > 0.7 且 BUY
    action = 2  # 强制 HOLD
```

---

### 与 4.7 的对比

| 维度 | Exp4.7 | Exp4.9 |
|------|--------|--------|
| revise_state 输入 | `s` (120d) | `s` (120d) + `regime_vector` (5d) |
| intrinsic_reward 逻辑 | 统一阈值 | 按机制分支 |
| Prompt 指导 | "生成通用技术指标" | "根据 regime 生成条件化特征和奖励" |
| COT 反馈 | 全局 Sharpe | 按机制分组的 Sharpe |
| Feature 分析 | 全局重要性 | 全局 + 按机制分组重要性 |
| DQN 网络 | 3 层 MLP | **不变** |
| DQN 超参 | 默认值 | **不变** |
| Baseline | 原始 OHLCV | **不变** |

### 文件结构

```
exp4.9/
├── regime_detector.py      # 新增: 市场机制检测 (纯规则)
├── prompts.py              # 重写: 机制感知 prompt
├── lesr_controller.py      # 增强: 适配新函数签名
├── dqn_trainer.py          # 微调: 只改 state 构建和 revise_state 调用
├── lesr_strategy.py        # 微调: 回测注入 regime + 安全网
├── feature_analyzer.py     # 增强: 按机制分析
├── baseline.py             # 复制: 不改
├── main.py                 # 新入口
├── config.yaml             # 新配置
└── README.md               # 本文档
```

### 实施步骤

| Phase | 任务 | 依赖 |
|-------|------|------|
| 1 | `regime_detector.py` 实现 + 单元测试 | 无 |
| 2 | `prompts.py` 重写 (regime-aware prompt) | Phase 1 |
| 3 | `dqn_trainer.py` 微调 (regime 注入) | Phase 1 |
| 4 | `lesr_controller.py` 适配 (新签名验证) | Phase 2, 3 |
| 5 | `feature_analyzer.py` 增强 (按机制分析) | Phase 1 |
| 6 | `lesr_strategy.py` 微调 (回测集成) | Phase 1 |
| 7 | `main.py` + `config.yaml` 入口 | Phase 4, 5, 6 |
| 8 | W3 (2018-2020训练/2023测试) 先跑一轮验证 | Phase 7 |

### 预期效果

基于实验数据分析：

| 场景 | 4.7 胜率 | 4.9 预期 | 改进原因 |
|------|---------|---------|---------|
| 温和牛市 (2015, 2023) | 100% | 100% | 无退化 |
| 平稳牛市 (2017) | 75% | 75-100% | regime 确认趋势，增强信心 |
| 震荡 (2018) | 50% | 75%+ | meanrev_signal 引导震荡策略 |
| COVID (2020) | 50% | 75%+ | crisis_signal 阻止盲目入场 |
| **整体** | **75%** | **85%+** | 核心改善在"输的 5 组" |

