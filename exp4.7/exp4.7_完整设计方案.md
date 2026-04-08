# Exp4.7 金融择时选股小实验 - 完整设计方案

> **创建日期**: 2026-04-07
> **目标**: 验证LESR框架（LLM驱动的状态表示优化）能否迁移到金融择时选股场景

---

## 一、实验目标

### 1.1 核心目标

验证LESR框架能否迁移到金融场景，实现：
- 从量价数据到金融因子的**自动化特征工程**
- 基于反馈的**迭代优化机制**
- 风险感知的交易决策

### 1.2 成功标准

| 指标 | 目标 | 说明 |
|------|------|------|
| **Sharpe Ratio** | > 基线 + 20% | 主指标，风险调整后收益 |
| **Max Drawdown** | < 30% | 约束指标，风险控制 |
| **链路完整性** | 3轮迭代完成 | 证明迭代机制有效 |

---

## 二、实验配置总览

| 维度 | 配置 |
|------|------|
| **股票** | TSLA（高波动）+ MSFT（稳健） |
| **时间范围** | 2018-2023（6年） |
| **训练集** | 2018-2020（3年，~750交易日） |
| **验证集** | 2021-2022（2年，~500交易日） |
| **测试集** | 2023（1年，~250交易日） |
| **输入窗口** | 20天OHLCV（120维） |
| **动作空间** | 离散：买入(0)/卖出(1)/持有(2) |
| **RL算法** | DQN |
| **迭代轮次** | 3轮 |
| **特征分析** | 相关性 + SHAP |
| **基线** | 纯MLP + 原始量价 |

---

## 三、架构设计

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Exp4.7 完整架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  第0轮：初始化                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Prompt: 金融任务描述 + 量价语义 + OHLCV格式          │    │
│  │ ↓                                                    │    │
│  │ LLM生成: revise_state() + intrinsic_reward()        │    │
│  │ ↓                                                    │    │
│  │ 代码解析 + 验证（维度检查、数值范围检查）            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  第N轮：迭代优化（N=0,1,2）                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. LLM采样6个候选特征函数                            │    │
│  │ ↓                                                    │    │
│  │ 2. DQN训练（自定义训练器）                            │    │
│  │    ├─ 训练集: 2018-2020                             │    │
│  │    ├─ 验证集: 2021-2022                             │    │
│  │    └─ 记录每个episode的state和future_return         │    │
│  │ ↓                                                    │    │
│  │ 3. 特征分析（替代Lipschitz）                         │    │
│  │    ├─ 相关性分析: 每个特征与未来收益的相关性         │    │
│  │    ├─ SHAP分析: 非线性重要性                         │    │
│  │    └─ 综合评分 = 0.5×相关 + 0.5×SHAP                │    │
│  │ ↓                                                    │    │
│  │ 4. COT反馈生成                                       │    │
│  │    ├─ 展示每个代码的Sharpe/MaxDD                     │    │
│  │    ├─ 识别高相关性特征                               │    │
│  │    └─ 让LLM分析原因并给出改进建议                    │    │
│  │ ↓                                                    │    │
│  │ 5. 构建下一轮Prompt                                  │    │
│  │    ├─ 融合历史代码和反馈                             │    │
│  │    └─ 要求LLM基于反馈改进                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  最终评估                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 最佳策略在测试集(2023)回测（FINSABER）                │    │
│  │ vs 基线（纯MLP）对比                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 集成方案：训练与回测分离

```
┌─────────────────────────────────────────────────────────────┐
│              训练与回测分离架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  阶段1: DQN训练（自定义训练器，不用FINSABER）                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ class DQNTrainer:                                   │    │
│  │   ├─ __init__(revise_state, intrinsic_reward)      │    │
│  │   ├─ train(train_data)  # 经验回放+Q网络更新        │    │
│  │   └─ evaluate(val_data)   # 验证集评估              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  阶段2: 回测评估（用FINSABER）                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ class LESRStrategy(BaseStrategyIso):                │    │
│  │   ├─ on_data(): DQN决策 + framework交易执行         │    │
│  │   └─ 不做训练，只做推理                              │    │
│  │                                                      │    │
│  │ framework = FINSABERFrameworkHelper()                │    │
│  │ framework.run(strategy)                              │    │
│  │ metrics = framework.evaluate(strategy)              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、状态空间设计

### 4.1 原始状态（120维）

```python
# 输入：20天OHLCV数据
s[0:19]   - 20天收盘价
s[20:39]  - 20天开盘价
s[40:59]  - 20天最高价
s[60:79]  - 20天最低价
s[80:99]  - 20天成交量
s[100:119] - 20天调整后收盘价
```

### 4.2 LLM任务：revise_state()

```python
def revise_state(raw_state):
    """
    输入: 原始120维量价数据
    输出: 原始120维 + 计算的金融因子

    可用的技术指标类型：
    - 趋势: SMA, EMA, MACD, 趋势强度
    - 动量: RSI, ROC, 随机指标, 动量比率
    - 波动率: 标准差, ATR, 布林带宽度
    - 成交量: OBV, 成交量比率, 价量关系
    """
    # LLM生成代码...
    return enhanced_state
```

### 4.3 LLM任务：intrinsic_reward()

```python
def intrinsic_reward(enhanced_state):
    """
    输入: 增强状态
    输出: [-100, 100]的风险判定

    正值: 状态适合交易（趋势明确、风险可控）
    负值: 状态不适合交易（震荡、高风险）

    必须使用至少一个新增的特征维度
    """
    # LLM生成代码...
    return reward_value
```

---

## 五、特征分析机制（替代Lipschitz）

### 5.1 为什么不用Lipschitz

| Lipschitz假设 | 金融数据现实 | 结果 |
|--------------|-------------|------|
| 状态空间连续 | 价格跳变、涨跌停 | ❌ 违反 |
| 奖励函数光滑 | 盈利瞬间跳变 | ❌ 违反 |
| 低噪声 | 高噪声（SNR≈8dB） | ❌ 违反 |
| 平稳过程 | 非平稳（分布漂移） | ❌ 违反 |

### 5.2 替代方案：相关性 + SHAP

```python
def analyze_features(episode_states, episode_rewards, original_dim):
    """
    替代Lipschitz的特征重要性分析

    Args:
        episode_states: 每个episode的增强状态列表
        episode_rewards: 每个episode的未来收益列表
        original_dim: 原始状态维度（120）

    Returns:
        importance: 综合重要性评分
        correlations: 各特征与收益的相关系数
        shap_values: SHAP重要性
    """
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestRegressor
    import shap

    states = np.array(episode_states)
    rewards = np.array(episode_rewards)

    # 只分析新增的特征维度
    extra_features = states[:, original_dim:]

    # 方法1: 相关性分析（Spearman，更鲁棒）
    correlations = []
    for i in range(extra_features.shape[1]):
        corr, p_value = spearmanr(extra_features[:, i], rewards)
        correlations.append(abs(corr))  # 取绝对值

    # 方法2: SHAP分析
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(extra_features, rewards)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(extra_features)
    shap_importance = np.abs(shap_values).mean(axis=0)

    # 综合评分
    importance = 0.5 * np.array(correlations) + 0.5 * shap_importance

    return importance, correlations, shap_importance
```

---

## 六、COT反馈生成（金融版）

### 6.1 反馈模板

```python
def get_financial_cot_prompt(codes, scores, importance, correlations, original_dim):
    """
    生成金融场景的COT反馈

    Args:
        codes: LLM生成的代码列表
        scores: 每个代码的性能指标（Sharpe, MaxDD等）
        importance: 特征重要性矩阵
        correlations: 特征相关性矩阵
        original_dim: 原始状态维度
    """
    s_feedback = ''

    for i, (code, score) in enumerate(zip(codes, scores)):
        s_feedback += f'========== 代码候选 -- {i+1} ==========\n'
        s_feedback += code + '\n'
        s_feedback += f'性能指标:\n'
        s_feedback += f'  夏普比率: {score["sharpe"]:.3f}\n'
        s_feedback += f'  最大回撤: {score["max_dd"]:.2f}%\n'
        s_feedback += f'  总收益: {score["total_return"]:.2f}%\n'

        # 原始特征分析
        s_feedback += f'\n原始特征(量价)的重要性:\n'
        for idx in range(min(5, original_dim)):
            s_feedback += f'  s[{idx}]: 重要性={importance[i][idx]:.3f}\n'

        # 新增特征分析
        extra_dim = importance[i].shape[0] - original_dim
        if extra_dim > 0:
            s_feedback += f'\n新增特征的重要性（Top3）:\n'
            top_extra = np.argsort(importance[i][original_dim:])[-3:][::-1]
            for rank, idx in enumerate(top_extra, 1):
                actual_idx = original_dim + idx
                s_feedback += f'  新增特征{idx}: 重要性={importance[i][actual_idx]:.3f}, 相关系数={correlations[i][actual_idx]:.3f}\n'

        s_feedback += '\n'

    cot_prompt = f"""
我们使用{len(codes)}个不同的状态表示和内在奖励函数组合训练了DQN策略。

训练结果：
{s_feedback}

请分析以上结果，并给出改进建议：

分析要点：
(a) 为什么某些代码表现更好？哪些特征贡献最大？
(b) 低性能代码的共同问题是什么？（如：特征冗余、过拟合、缺少关键信号）
(c) 如何改进状态表示和内在奖励设计？

金融场景特别提示：
- 趋势特征（动量、均线）对择时很重要
- 波动率特征有助于风险控制
- 成交量可以确认价格趋势
- intrinsic_reward应该在趋势明确时给正值，震荡时给负值

目标：提高策略的夏普比率，同时控制最大回撤在30%以内。
"""
    return cot_prompt
```

---

## 七、初始化Prompt模板

### 7.1 第0轮初始化Prompt

```python
INITIAL_PROMPT = """
你是金融量化分析专家，擅长从量价数据中提取交易信号。

## 任务背景

使用强化学习训练股票择时策略。策略在每个交易日决定：
- 买入(BUY): 建立多头仓位
- 卖出(SELL): 平仓
- 持有(HOLD): 保持当前仓位

## 可用数据

原始状态（120维NumPy数组）:
- s[0:19]: 20天收盘价序列
- s[20:39]: 20天开盘价序列
- s[40:59]: 20天最高价序列
- s[60:79]: 20天最低价序列
- s[80:99]: 20天成交量序列
- s[100:119]: 20天调整后收盘价序列

## 金融语义说明

**关键概念**:

1. **收益率**: 价格变化的百分比
   - 日收益率 = (今日收盘价 - 昨日收盘价) / 昨日收盘价

2. **波动率**: 价格变动的剧烈程度
   - 计算方式：收益率的标准差
   - 金融含义：风险的度量

3. **趋势**: 价格的持续运动方向
   - 上升趋势：短期均线 > 长期均线
   - 下降趋势：短期均线 < 长期均线
   - 震荡趋势：价格在区间内波动

4. **动量**: 价格变化的速度和方向
   - 正动量：价格上涨加速
   - 负动量：价格下跌加速

5. **支撑/阻力**: 价格难以突破的水平
   - 支撑位：价格下跌时的"地板"
   - 阻力位：价格上涨时的"天花板"

## 目标函数

策略的绩效由以下指标衡量:
1. **夏普比率**: 风险调整后收益（最大化）
2. **最大回撤**: 最大损失幅度（最小化，控制在30%以内）
3. **总收益**: 累积收益率（最大化）

## 约束条件

1. 交易成本：每笔交易收取0.1%佣金
2. 仓位限制：单只股票最大仓位100%
3. 风险限制：单日最大亏损不超过5%

## 你的任务

请生成两个Python函数：

### 函数1: revise_state(raw_state)
- 输入: 原始状态（120维NumPy数组）
- 输出: 增强状态（原始120维 + 新增特征）
- 建议：
  - 可以计算的技术指标：SMA, EMA, RSI, MACD, 布林带, ATR等
  - 可以计算多时间尺度特征（5日、10日、20日）
  - 处理边界情况（除零、空值等）

### 函数2: intrinsic_reward(enhanced_state)
- 输入: 增强状态
- 输出: 内在奖励值（范围[-100, 100]）
- 建议：
  - 正值：当前状态适合交易（趋势明确、风险可控）
  - 负值：当前状态不适合交易（震荡、高风险）
  - 必须使用至少一个新增的特征维度

## 输出格式

请返回完整的、可执行的Python代码：

```python
import numpy as np

def revise_state(s):
    # 你的实现
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # 你的实现
    return reward
```

Let's think step by step.
"""
```

---

## 八、算法选择说明

### 8.1 算法分类

```
强化学习算法分类：
├── 经典RL (表格式)
│   └── Q-Learning, SARSA
│
└── 深度RL (DRL, 用神经网络逼近)
    ├── DQN (Deep Q-Network) ← 我们选的
    ├── TD3 (Twin Delayed DDPG)
    ├── PPO (Proximal Policy Optimization)
    └── ...
```

### 8.2 算法选择

| 组件 | 选择 | 理由 |
|------|------|------|
| **RL算法** | **DQN** (深度强化学习) | 动作空间是离散的（买/卖/持有） |
| **神经网络** | **MLP** (2层隐藏层, 256维) | 简单有效，适合表格型数据 |
| **经验回放** | **ReplayBuffer** (容量10000) | 打破样本相关性，稳定训练 |
| **目标网络** | **软更新** (τ=0.005) | 稳定Q值估计 |
| **优化器** | **Adam** (lr=1e-3) | 自适应学习率 |
| **探索策略** | **ε-greedy** (1.0→0.1) | 平衡探索与利用 |

### 8.3 与LESR原版对比

| 项目 | LESR原版 | 我们的金融版本 | 原因 |
|------|---------|---------------|------|
| RL算法 | **TD3** (连续动作) | **DQN** (离散动作) | 金融择时是离散决策 |
| 动作空间 | 连续 (关节力矩) | 离散 (买/卖/持有) | 匹配任务特性 |
| 都是DRL | ✅ | ✅ | 都用神经网络近似 |
| 适用场景 | 机器人连续控制 | 金融离散决策 | 任务不同 |

### 8.4 为什么选DQN？

1. **动作空间匹配** - 金融择时本质是三元离散决策，不是连续控制
2. **训练稳定性** - DQN在离散动作空间比TD3/PPO更稳定
3. **基线对比公平** - 纯MLP基线也用DQN，控制变量
4. **实现简洁** - 对于小实验，DQN足够验证假设

---

## 九、DQN训练器设计

### 9.1 DQNTrainer类

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim=3, hidden_dim=256):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()


class DQNTrainer:
    def __init__(self, ticker, revise_state_func, intrinsic_reward_func,
                 state_dim, intrinsic_weight=0.02):
        self.ticker = ticker
        self.revise_state = revise_state_func
        self.intrinsic_reward = intrinsic_reward_func
        self.intrinsic_weight = intrinsic_weight

        self.dqn = DQN(state_dim)
        self.target_dqn = DQN(state_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()

        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        # 用于特征分析
        self.episode_states = []
        self.episode_rewards = []

    def calculate_reward(self, current_price, next_price):
        """计算交易奖励（基于价格变化）"""
        return (next_price - current_price) / current_price

    def extract_state(self, data_loader, date, window=20):
        """提取20天窗口的状态"""
        dates = data_loader.get_date_range()
        try:
            idx = dates.index(date)
        except ValueError:
            return None

        if idx < window - 1:
            return None

        window_dates = dates[idx-window+1:idx+1]
        state_120d = []

        for d in window_dates:
            daily_data = data_loader.get_data_by_date(d)
            if self.ticker in daily_data['price']:
                price_dict = daily_data['price'][self.ticker]
                state_120d.extend([
                    price_dict.get('close', 0),
                    price_dict.get('open', 0),
                    price_dict.get('high', 0),
                    price_dict.get('low', 0),
                    price_dict.get('volume', 0),
                    price_dict.get('adjusted_close', 0)
                ])

        return np.array(state_120d)

    def train(self, train_data_loader, start_date, end_date, max_episodes=100):
        """训练DQN"""
        dates = [d for d in train_data_loader.get_date_range()
                 if start_date <= d <= end_date]

        total_reward = 0
        for episode in range(max_episodes):
            state = None
            epsilon = max(self.epsilon_end, self.epsilon * (self.epsilon_decay ** episode))

            for i, date in enumerate(dates):
                raw_state = self.extract_state(train_data_loader, date)
                if raw_state is None:
                    continue

                # 特征增强
                enhanced_state = self.revise_state(raw_state)

                # 选择动作
                action = self.dqn.select_action(enhanced_state, epsilon)

                # 计算奖励
                current_price = train_data_loader.get_ticker_price_by_date(self.ticker, date)
                if i < len(dates) - 1:
                    next_date = dates[i + 1]
                    next_price = train_data_loader.get_ticker_price_by_date(self.ticker, next_date)
                    reward = self.calculate_reward(current_price, next_price)

                    # 添加内在奖励
                    intrinsic_r = self.intrinsic_reward(enhanced_state)
                    total_reward = reward + self.intrinsic_weight * intrinsic_r

                    # 获取下一状态
                    next_raw_state = self.extract_state(train_data_loader, next_date)
                    if next_raw_state is not None:
                        next_enhanced = self.revise_state(next_raw_state)
                    else:
                        next_enhanced = enhanced_state
                        done = True
                    done = False
                else:
                    next_enhanced = enhanced_state
                    total_reward = 0
                    done = True

                # 存储经验
                self.buffer.push(enhanced_state, action, total_reward, next_enhanced, done)

                # 记录用于特征分析
                self.episode_states.append(enhanced_state.copy())
                self.episode_rewards.append(total_reward)

                # 训练网络
                if len(self.buffer) > self.batch_size:
                    self._update_network()

                if done:
                    break

            # 软更新目标网络
            self._soft_update_target()

        return self._get_training_summary()

    def _update_network(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_dqn(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _soft_update_target(self):
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _get_training_summary(self):
        """返回训练摘要，用于特征分析"""
        return {
            'states': self.episode_states,
            'rewards': self.episode_rewards
        }

    def evaluate(self, val_data_loader, start_date, end_date):
        """在验证集上评估"""
        dates = [d for d in val_data_loader.get_date_range()
                 if start_date <= d <= end_date]

        total_return = 0
        trades = []
        current_position = 0

        for date in dates:
            raw_state = self.extract_state(val_data_loader, date)
            if raw_state is None:
                continue

            enhanced_state = self.revise_state(raw_state)
            action = self.dqn.select_action(enhanced_state, epsilon=0.0)

            current_price = val_data_loader.get_ticker_price_by_date(self.ticker, date)

            if action == 0:  # 买入
                if current_position == 0:
                    current_position = 1
                    trades.append(('buy', date, current_price))
            elif action == 1:  # 卖出
                if current_position == 1:
                    current_position = 0
                    trades.append(('sell', date, current_price))

            # 计算持有收益
            if current_position == 1 and len(trades) > 0:
                entry_price = trades[-1][2]
                total_return += (current_price - entry_price) / entry_price

        # 计算评估指标
        sharpe = self._calculate_sharpe(trades, val_data_loader)
        max_dd = self._calculate_max_drawdown(trades, val_data_loader)

        return {
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_return': total_return * 100,
            'trades': trades
        }
```

---

## 十、FINSABER回测集成

### 9.1 LESRStrategy（推理模式）

```python
from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper
import numpy as np

class LESRStrategy(BaseStrategyIso):
    def __init__(self, ticker, revise_state_func, trained_dqn):
        super().__init__()
        self.ticker = ticker
        self.revise_state = revise_state_func
        self.dqn = trained_dqn
        self.window = 20

    def on_data(self, date, data_loader, framework):
        # 提取20天历史数据
        raw_state = self._extract_20day_window(date, data_loader)
        if raw_state is None:
            return

        # LLM特征增强
        enhanced_state = self.revise_state(raw_state)

        # DQN决策（评估模式，不探索）
        action = self.dqn.select_action(enhanced_state, epsilon=0.0)

        # 获取当前价格
        current_price = data_loader.get_ticker_price_by_date(self.ticker, date)

        # 执行交易
        if action == 0:  # 买入
            framework.buy(date, self.ticker, current_price, -1)
            self.logger.info(f"BUY {self.ticker} at {current_price:.2f} on {date}")
        elif action == 1:  # 卖出
            if self.ticker in framework.portfolio:
                framework.sell(date, self.ticker, current_price,
                              framework.portfolio[self.ticker]['quantity'])
                self.logger.info(f"SELL {self.ticker} at {current_price:.2f} on {date}")
        # action == 2: 持有，不做操作

    def _extract_20day_window(self, date, data_loader):
        """提取20天OHLCV数据作为原始状态"""
        dates = data_loader.get_date_range()
        try:
            idx = dates.index(date)
        except ValueError:
            return None

        if idx < self.window - 1:
            return None

        window_dates = dates[idx-self.window+1:idx+1]
        state_120d = []

        for d in window_dates:
            daily_data = data_loader.get_data_by_date(d)
            if self.ticker in daily_data['price']:
                price_dict = daily_data['price'][self.ticker]
                state_120d.extend([
                    price_dict.get('close', 0),
                    price_dict.get('open', 0),
                    price_dict.get('high', 0),
                    price_dict.get('low', 0),
                    price_dict.get('volume', 0),
                    price_dict.get('adjusted_close', 0)
                ])

        return np.array(state_120d)
```

### 9.2 基线策略

```python
class BaselineMLPStrategy(BaseStrategyIso):
    def __init__(self, ticker, mlp_model):
        super().__init__()
        self.ticker = ticker
        self.mlp = mlp_model
        self.window = 20

    def on_data(self, date, data_loader, framework):
        # 直接用原始120维，不做任何特征工程
        raw_state = self._extract_20day_window(date, data_loader)
        if raw_state is None:
            return

        action = self.mlp.select_action(raw_state, epsilon=0.0)

        current_price = data_loader.get_ticker_price_by_date(self.ticker, date)

        if action == 0:  # 买入
            framework.buy(date, self.ticker, current_price, -1)
        elif action == 1:  # 卖出
            if self.ticker in framework.portfolio:
                framework.sell(date, self.ticker, current_price,
                              framework.portfolio[self.ticker]['quantity'])

    def _extract_20day_window(self, date, data_loader):
        """与LESRStrategy相同的窗口提取逻辑"""
        dates = data_loader.get_date_range()
        try:
            idx = dates.index(date)
        except ValueError:
            return None

        if idx < self.window - 1:
            return None

        window_dates = dates[idx-self.window+1:idx+1]
        state_120d = []

        for d in window_dates:
            daily_data = data_loader.get_data_by_date(d)
            if self.ticker in daily_data['price']:
                price_dict = daily_data['price'][self.ticker]
                state_120d.extend([
                    price_dict.get('close', 0),
                    price_dict.get('open', 0),
                    price_dict.get('high', 0),
                    price_dict.get('low', 0),
                    price_dict.get('volume', 0),
                    price_dict.get('adjusted_close', 0)
                ])

        return np.array(state_120d)
```

---

## 十一、LESR迭代控制器

### 10.1 LESRController类

```python
import os
import openai
import importlib
import numpy as np
from typing import List, Dict, Tuple
import pickle

class LESRController:
    def __init__(self, config):
        self.tickers = config['tickers']  # ['TSLA', 'MSFT']
        self.train_period = config['train_period']  # ('2018-01-01', '2020-12-31')
        self.val_period = config['val_period']  # ('2021-01-01', '2022-12-31')
        self.test_period = config['test_period']  # ('2023-01-01', '2023-12-31')
        self.data_loader = config['data_loader']
        self.sample_count = config.get('sample_count', 6)
        self.max_iterations = config.get('max_iterations', 3)
        self.openai_key = config['openai_key']
        self.model = config.get('model', 'gpt-4')
        self.output_dir = config.get('output_dir', 'exp4.7/results')

        openai.api_key = self.openai_key

        # 历史记录
        self.all_iter_results = []
        self.all_iter_cot_suggestions = []
        self.all_codes = []

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def run_optimization(self):
        """主优化循环"""
        for iteration in range(self.max_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration}")
            print(f"{'='*50}")

            # 1. 生成Prompt
            if iteration == 0:
                prompt = self._get_initial_prompt()
            else:
                prompt = self._get_iteration_prompt()

            # 2. LLM采样
            valid_samples = self._sample_functions(prompt, iteration)

            if len(valid_samples) == 0:
                print("No valid samples generated, skipping iteration")
                continue

            self.all_codes.append(valid_samples)

            # 3. 并行训练
            results = self._parallel_train(valid_samples)

            # 4. 特征分析
            analysis = self._analyze_results(valid_samples, results)

            # 5. COT反馈
            cot_suggestion = self._generate_cot_feedback(valid_samples, results, analysis)
            self.all_iter_cot_suggestions.append(cot_suggestion)

            # 6. 保存结果
            self._save_iteration_results(iteration, valid_samples, results, analysis)

        # 7. 选择最佳策略
        best_strategy = self._select_best_strategy()

        return best_strategy

    def _get_initial_prompt(self):
        """生成初始化Prompt"""
        return INITIAL_PROMPT

    def _get_iteration_prompt(self):
        """生成迭代Prompt"""
        # 构建历史经验
        former_history = ''
        for i in range(len(self.all_codes)):
            former_history += f'\n\n\nFormer Iteration:{i + 1}\n'
            for j, code in enumerate(self.all_codes[i]):
                former_history += f'候选{j+1}:\n{code}\n'
            former_history += f'\n建议:\n{self.all_iter_cot_suggestions[i]}\n'

        iteration_prompt = f"""
你是金融量化分析专家。

我们已经进行了多轮迭代，以下是历史经验：

{former_history}

基于以上经验和建议，请生成改进的状态表示和内在奖励函数。

要求：
1. 避免重复已经验证无效的特征
2. 保留并改进有效的特征
3. 尝试新的特征组合
4. intrinsic_reward必须在[-100, 100]范围内

请返回完整的Python代码。
"""
        return iteration_prompt

    def _sample_functions(self, prompt, iteration):
        """从LLM采样多个候选函数"""
        valid_samples = []

        for sample_id in range(self.sample_count):
            print(f"Sampling {sample_id + 1}/{self.sample_count}...")

            try:
                # 调用LLM
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是金融量化分析专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000
                )

                code = response['choices'][0]['message']['content']

                # 提取Python代码
                code = self._extract_python_code(code)

                # 保存并验证
                code_path = os.path.join(self.output_dir, f'it{iteration}_sample{sample_id}.py')
                with open(code_path, 'w') as f:
                    f.write(code)

                # 动态导入验证
                module = importlib.import_module(f'exp4.7.results.it{iteration}_sample{sample_id}')

                # 测试函数
                test_state = np.zeros(120)
                enhanced = module.revise_state(test_state)
                intrinsic_r = module.intrinsic_reward(enhanced)

                # 验证
                assert enhanced.shape[0] >= 120, "输出维度必须 >= 120"
                assert -100 <= intrinsic_r <= 100, f"intrinsic_reward必须在[-100, 100]，实际: {intrinsic_r}"

                valid_samples.append({
                    'code': code,
                    'module': module,
                    'state_dim': enhanced.shape[0],
                    'original_dim': 120
                })

                print(f"  Sample {sample_id + 1} validated: state_dim={enhanced.shape[0]}")

            except Exception as e:
                print(f"  Sample {sample_id + 1} failed: {e}")
                continue

        return valid_samples

    def _extract_python_code(self, text):
        """从LLM输出中提取Python代码"""
        # 查找import numpy语句
        if 'import numpy' in text:
            start = text.index('import numpy')
        else:
            raise ValueError("No import numpy found")

        # 查找最后的return
        end = text.rindex('return')
        while end < len(text) and text[end] != '\n':
            end += 1

        code = text[start:end + 1]
        return code

    def _parallel_train(self, samples):
        """并行训练多个候选"""
        results = []

        for i, sample in enumerate(samples):
            print(f"\nTraining sample {i + 1}/{len(samples)}...")

            for ticker in self.tickers:
                try:
                    trainer = DQNTrainer(
                        ticker=ticker,
                        revise_state_func=sample['module'].revise_state,
                        intrinsic_reward_func=sample['module'].intrinsic_reward,
                        state_dim=sample['state_dim']
                    )

                    # 训练
                    print(f"  Training on {ticker}...")
                    trainer.train(
                        self.data_loader,
                        self.train_period[0],
                        self.train_period[1],
                        max_episodes=50
                    )

                    # 验证
                    print(f"  Evaluating on {ticker}...")
                    val_metrics = trainer.evaluate(
                        self.data_loader,
                        self.val_period[0],
                        self.val_period[1]
                    )

                    results.append({
                        'sample_id': i,
                        'ticker': ticker,
                        'sharpe': val_metrics['sharpe'],
                        'max_dd': val_metrics['max_dd'],
                        'total_return': val_metrics['total_return'],
                        'trainer': trainer
                    })

                    print(f"  Sharpe: {val_metrics['sharpe']:.3f}, MaxDD: {val_metrics['max_dd']:.2f}%")

                except Exception as e:
                    print(f"  Training failed: {e}")
                    continue

        return results

    def _analyze_results(self, samples, results):
        """分析结果，生成特征重要性"""
        analysis = []

        for i, sample in enumerate(samples):
            sample_results = [r for r in results if r['sample_id'] == i]

            if len(sample_results) == 0:
                continue

            # 收集所有该样本的训练数据
            all_states = []
            all_rewards = []

            for result in sample_results:
                summary = result['trainer']._get_training_summary()
                all_states.extend(summary['states'])
                all_rewards.extend(summary['rewards'])

            # 特征分析
            importance, correlations, shap_values = analyze_features(
                all_states,
                all_rewards,
                sample['original_dim']
            )

            analysis.append({
                'sample_id': i,
                'importance': importance,
                'correlations': correlations,
                'shap_values': shap_values
            })

        return analysis

    def _generate_cot_feedback(self, samples, results, analysis):
        """生成COT反馈"""
        codes = [s['code'] for s in samples]
        scores = []

        for i, sample in enumerate(samples):
            sample_results = [r for r in results if r['sample_id'] == i]
            if len(sample_results) > 0:
                # 平均该样本在所有股票上的表现
                avg_sharpe = np.mean([r['sharpe'] for r in sample_results])
                avg_max_dd = np.mean([r['max_dd'] for r in sample_results])
                avg_return = np.mean([r['total_return'] for r in sample_results])
                scores.append({
                    'sharpe': avg_sharpe,
                    'max_dd': avg_max_dd,
                    'total_return': avg_return
                })
            else:
                scores.append({'sharpe': 0, 'max_dd': 100, 'total_return': 0})

        importance_list = [a['importance'] for a in analysis]
        correlations_list = [a['correlations'] for a in analysis]

        return get_financial_cot_feedback(
            codes, scores, importance_list, correlations_list, 120
        )

    def _save_iteration_results(self, iteration, samples, results, analysis):
        """保存迭代结果"""
        iteration_dir = os.path.join(self.output_dir, f'iteration_{iteration}')
        os.makedirs(iteration_dir, exist_ok=True)

        # 保存结果
        with open(os.path.join(iteration_dir, 'results.pkl'), 'wb') as f:
            pickle.dump({
                'samples': samples,
                'results': results,
                'analysis': analysis
            }, f)

    def _select_best_strategy(self):
        """选择最佳策略"""
        # 从所有迭代中找最佳
        best_sharpe = -float('inf')
        best_config = None

        for iteration in range(self.max_iterations):
            iteration_dir = os.path.join(self.output_dir, f'iteration_{iteration}')
            result_file = os.path.join(iteration_dir, 'results.pkl')

            if os.path.exists(result_file):
                with open(result_file, 'rb') as f:
                    data = pickle.load(f)

                for result in data['results']:
                    if result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_config = {
                            'iteration': iteration,
                            'sample_id': result['sample_id'],
                            'ticker': result['ticker'],
                            'sharpe': result['sharpe'],
                            'trainer': result['trainer']
                        }

        print(f"\nBest strategy: Iteration {best_config['iteration']}, "
              f"Sample {best_config['sample_id']}, Sharpe = {best_config['sharpe']:.3f}")

        return best_config
```

---

## 十二、基线对比

### 11.1 基线DQN训练

```python
class BaselineDQN(nn.Module):
    """基线：直接用原始120维量价，不做特征工程"""
    def __init__(self, action_dim=3, hidden_dim=256):
        super(BaselineDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(120, hidden_dim),  # 直接用120维
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()


def train_baseline(ticker, data_loader, train_period, val_period):
    """训练基线模型"""
    # 使用identity函数作为revise_state（不做特征工程）
    def identity_revise_state(raw_state):
        return raw_state  # 直接返回，不添加特征

    def zero_intrinsic_reward(state):
        return 0.0  # 不使用内在奖励

    trainer = DQNTrainer(
        ticker=ticker,
        revise_state_func=identity_revise_state,
        intrinsic_reward_func=zero_intrinsic_reward,
        state_dim=120,
        intrinsic_weight=0.0
    )

    # 训练
    trainer.train(data_loader, train_period[0], train_period[1], max_episodes=50)

    # 评估
    val_metrics = trainer.evaluate(data_loader, val_period[0], val_period[1])

    return trainer, val_metrics
```

---

## 十三、完整实验流程

### 12.1 主入口

```python
import pickle
from backtest.data_util import FinMemDataset
from backtest.finsaber import FINSABER
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper

def main():
    # 1. 加载数据
    data_loader = FinMemDataset(
        pickle_file="data/finmem_data/stock_data_cherrypick_2000_2024.pkl"
    )

    # 2. LESR优化
    config = {
        'tickers': ['TSLA', 'MSFT'],
        'train_period': ('2018-01-01', '2020-12-31'),
        'val_period': ('2021-01-01', '2022-12-31'),
        'test_period': ('2023-01-01', '2023-12-31'),
        'data_loader': data_loader,
        'sample_count': 6,
        'max_iterations': 3,
        'openai_key': os.getenv('OPENAI_API_KEY'),
        'model': 'gpt-4',
        'output_dir': 'exp4.7/results'
    }

    controller = LESRController(config)
    best_config = controller.run_optimization()

    # 3. 测试集回测（FINSABER）
    print("\n" + "="*50)
    print("Testing on 2023 data...")
    print("="*50)

    best_trainer = best_config['trainer']
    best_ticker = best_config['ticker']

    lesl_strategy = LESRStrategy(
        ticker=best_ticker,
        revise_state_func=best_trainer.revise_state,
        trained_dqn=best_trainer.dqn
    )

    framework = FINSABERFrameworkHelper(initial_cash=100000)
    framework.load_backtest_data_single_ticker(
        data_loader,
        best_ticker,
        start_date=pd.to_datetime(config['test_period'][0]),
        end_date=pd.to_datetime(config['test_period'][1])
    )

    framework.run(lesr_strategy)
    lesl_metrics = framework.evaluate(lesr_strategy)

    print(f"\nLESR Results on Test Set:")
    print(f"  Sharpe: {lesl_metrics['sharpe_ratio']:.3f}")
    print(f"  Max DD: {lesl_metrics['max_drawdown']:.2f}%")
    print(f"  Total Return: {lesl_metrics['total_return']*100:.2f}%")

    # 4. 基线对比
    baseline_trainer, baseline_val = train_baseline(
        best_ticker, data_loader, config['train_period'], config['val_period']
    )

    baseline_strategy = BaselineMLPStrategy(best_ticker, baseline_trainer.dqn)

    framework.reset()
    framework.load_backtest_data_single_ticker(
        data_loader,
        best_ticker,
        start_date=pd.to_datetime(config['test_period'][0]),
        end_date=pd.to_datetime(config['test_period'][1])
    )

    framework.run(baseline_strategy)
    baseline_metrics = framework.evaluate(baseline_strategy)

    print(f"\nBaseline Results on Test Set:")
    print(f"  Sharpe: {baseline_metrics['sharpe_ratio']:.3f}")
    print(f"  Max DD: {baseline_metrics['max_drawdown']:.2f}%")
    print(f"  Total Return: {baseline_metrics['total_return']*100:.2f}%")

    # 5. 对比结果
    print(f"\n{'='*50}")
    print("Final Comparison:")
    print(f"{'='*50}")
    print(f"LESR Sharpe: {lesl_metrics['sharpe_ratio']:.3f}")
    print(f"Baseline Sharpe: {baseline_metrics['sharpe_ratio']:.3f}")
    print(f"Improvement: {(lesl_metrics['sharpe_ratio']/baseline_metrics['sharpe_ratio']-1)*100:.1f}%")

    # 6. 保存结果
    results = {
        'lesr': lesl_metrics,
        'baseline': baseline_metrics,
        'config': config
    }

    with open('exp4.7/results/final_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
```

---

## 十四、项目目录结构

```
exp4.7/
├── README.md                              # 本文档
├── exp4.7_完整设计方案.md                  # 本文档
├── lesr_strategy.py                       # LESR策略实现
├── dqn_trainer.py                         # DQN训练器
├── lesr_controller.py                     # LESR迭代控制器
├── baseline.py                            # 基线实现
├── feature_analyzer.py                    # 特征分析模块
├── prompts.py                             # Prompt模板
├── main.py                                # 主入口
├── config.yaml                            # 配置文件
└── results/                               # 结果输出目录
    ├── iteration_0/
    ├── iteration_1/
    ├── iteration_2/
    └── final_results.pkl
```

---

## 十五、预期时间表

| 阶段 | 任务 | 时间 |
|------|------|------|
| Week 1-2 | 数据准备 + DQN训练器 | 2周 |
| Week 3 | LESR迭代控制器 | 1周 |
| Week 4-5 | 运行实验 | 2周 |
| Week 6 | 分析和报告 | 1周 |

---

## 十六、风险评估

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| LLM生成代码质量差 | 中 | 高 | 增强验证、多采样 |
| DQN训练不稳定 | 中 | 中 | 调整超参数、增加训练轮数 |
| 特征分析失效 | 低 | 中 | 多方法交叉验证 |
| 过拟合验证集 | 中 | 高 | 测试集最终验证 |

---

**文档版本**: v1.0
**最后更新**: 2026-04-07
**状态**: 待评审
