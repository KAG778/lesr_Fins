# LESR 系统架构与功能模块分析

## 项目概述

**LESR (LLM-Empowered State Representation for Reinforcement Learning)** 是一个利用大语言模型(LLM)自动生成任务相关状态表示代码，从而增强强化学习训练效果的 novel 框架。

### 核心创新
- **LLM 驱动的状态表示优化**：利用 LLM 自动生成 `revise_state` 和 `intrinsic_reward` 函数
- **迭代式改进**：通过多轮迭代和反馈机制不断优化状态表示
- **Lipschitz 常数分析**：基于状态-奖励映射的连续性分析提供反馈

---

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LESR 主循环控制                                    │
│                        (lesr_main.py)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌──────────────────┐ ┌──────────────┐ ┌─────────────────┐
        │  LLM 交互模块     │ │  训练管理     │ │  结果评估模块    │
        │  (OpenAI API)    │ │  (Tmux并发)  │ │  (Policy评估)   │
        └──────────────────┘ └──────────────┘ └─────────────────┘
                    │               │               │
                    ▼               ▼               ▼
        ┌──────────────────┐ ┌──────────────┐ ┌─────────────────┐
        │ Prompt 模板引擎   │ │ lesr_train.py│ │  状态分析模块    │
        │ - 初始化Prompt   │ │              │ │ - Lipschitz计算 │
        │ - COT反馈Prompt  │ │ - TD3算法    │ │ - 相关性分析     │
        │ - 迭代Prompt     │ │ - 环境交互    │ │                 │
        └──────────────────┘ └──────────────┘ └─────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌──────────────────┐ ┌──────────────┐ ┌─────────────────┐
        │  强化学习核心     │ │  环境适配层   │ │  工具模块        │
        │  - TD3.py        │ │  - Gymnasium │ │  - utils.py     │
        │  - Actor/Critic  │ │  - MuJoCo    │ │  - ReplayBuffer │
        │  - 目标网络更新   │ │  - Robotics  │ │                 │
        └──────────────────┘ └──────────────┘ └─────────────────┘
```

---

## 核心功能模块详解

### 1. 主控制模块 (lesr_main.py)

**功能职责**：协调整个 LESR 训练流程的主循环

#### 1.1 初始化阶段
```
init_prompt()
├── 读取环境观察空间描述 (Excel)
├── 生成初始 Prompt 模板
├── 初始化 OpenAI API 连接
└── 创建输出目录结构
```

#### 1.2 迭代训练循环
```python
for iteration in range(args.iteration):
    # 1. LLM 采样阶段
    sample_state_revision_functions()
    
    # 2. 并行训练阶段
    parallel_training_with_tmux()
    
    # 3. 等待训练完成
    wait_for_training_completion()
    
    # 4. 结果分析阶段
    analyze_results_with_lipschitz()
    
    # 5. COT 反馈生成
    generate_cot_feedback()
    
    # 6. 更新 Prompt 进行下一轮
    update_prompt_for_next_iteration()
```

#### 1.3 最终评估阶段
```
evaluate_best_policy()
├── 加载最佳状态表示函数
├── 多种子评估
└── 生成最终性能报告
```

---

### 2. LLM 交互模块

#### 2.1 Prompt 生成系统

**初始化 Prompt 结构**：
```python
init_prompt_template = f"""
任务描述: {task_description}
状态空间: {total_dim} 维数组 s
状态细节: {detail_content}
目标: 设计 revise_state 和 intrinsic_reward 函数

约束条件:
1. 使用源状态维度 s[0] ~ s[{total_dim-1}]
2. 可以添加额外计算维度
3. intrinsic_reward 必须使用额外维度
{additional_prompt}
"""
```

**COT 反馈 Prompt 结构**：
```python
cot_prompt = f"""
已训练 {sample_count} 个不同的状态修订代码
监控指标:
1. 最终策略性能 (累积奖励)
2. 每个状态维度的 Lipschitz 常数

结果分析:
{s_feedback}

改进建议:
(a) 分析低性能代码失败原因
(b) 识别高相关性维度特征
(c) 提出改进方案
"""
```

**迭代 Prompt 结构**：
```python
next_iteration_prompt = f"""
历史经验:
{former_histoy}

历史建议:
{former_suggestions}

基于以上信息，生成改进的状态表示代码...
"""
```

#### 2.2 代码解析与验证流程
```
LLM 输出
    ↓
提取 Python 代码 (import 到 return)
    ↓
保存到临时文件
    ↓
动态导入模块
    ↓
功能测试
    ├── revise_state() 输出维度检查
    └── intrinsic_reward() 输出范围检查 [-100, 100]
    ↓
添加到有效样本池
```

---

### 3. 训练管理模块 (lesr_train.py)

#### 3.1 核心训练循环
```python
for t in range(max_timesteps):
    # 1. 状态修订
    revised_state = revise_state(original_state)
    
    # 2. 动作选择
    if t < start_timesteps:
        action = random_action()
    else:
        action = policy.select_action(revised_state) + noise
    
    # 3. 环境交互
    next_state, reward, done, info = env.step(action)
    
    # 4. 内在奖励计算
    intrinsic_r = intrinsic_w * intrinsic_reward(revised_state)
    
    # 5. 经验存储
    replay_buffer.add(revised_state, action, 
                      revised_next_state, 
                      reward + intrinsic_r, done)
    
    # 6. 策略训练
    if t >= start_timesteps:
        policy.train(replay_buffer, batch_size)
    
    # 7. Episode 结束处理
    if done:
        calculate_lipschitz_constants()
        update_state_correlation()
```

#### 3.2 Lipschitz 常数计算
```python
def cal_lipschitz(state_change, reward_change):
    """
    计算每个状态维度与奖励的 Lipschitz 常数
    用于评估状态-奖励映射的连续性
    """
    lipschitz = np.zeros([state_dim, ])
    for ii in range(state_dim):
        # 按状态变化排序
        cur_index = np.argsort(state_change[ii])
        
        # 计算相邻样本的奖励变化率
        cur_lipschitz = |reward[:-1] - reward[1]| / 
                        |state[:-1] - state[1]| + 1e-2
        
        # 取最大值作为该维度的 Lipschitz 常数
        lipschitz[ii] = cur_lipschitz.max()
    
    return lipschitz
```

#### 3.3 状态相关性软更新
```python
# 每个 episode 结束后更新
soft_state_correlation = corr_tau * state_lipschitz_constant_correlation + 
                        (1 - corr_tau) * soft_state_correlation
```

---

### 4. 强化学习核心模块 (TD3.py)

#### 4.1 网络架构
```
Actor 网络:
state → Linear(256) → ReLU → Linear(256) → ReLU → Linear(action_dim) → tanh × max_action

Critic 网络 (双 Q 网络):
state + action → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1) → Q1
                    ↓
                    └→ Linear(256) → ReLU → Linear(256) → ReLU → Linear(1) → Q2
```

#### 4.2 TD3 算法关键特性
1. **Twin Delayed DDPG**
   - 双 Critic 网络，取最小 Q 值
   - 延迟策略更新 (每 2 次 Critic 更新更新 1 次 Actor)

2. **目标策略平滑**
   - 添加噪声到目标动作
   - 剪切噪声到指定范围

3. **软目标更新**
   ```python
   target_param = τ * param + (1 - τ) * target_param
   τ = 0.005
   ```

---

### 5. 环境适配层

#### 5.1 支持的环境类型
```python
# Gymnasium 环境检测
if type(env.observation_space) == gym.spaces.Dict:
    # Robotics 任务 (AntMaze, Fetch, Adroit)
    state_dim = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
else:
    # MuJoCo 连续控制任务
    state_dim = obs.shape[0]
```

#### 5.2 任务特殊处理
```python
# 成功率评估任务
if 'antmaze' in env or 'fetch' in env or 'adroit' in env:
    eval_episodes = 50
    success_metric = True
    # 特殊终止条件
    if 'fetch' in env and abs(reward) < 0.05:
        terminated = True
```

---

### 6. 工具模块 (utils.py)

#### 6.1 经验回放缓冲区
```python
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1e6):
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
    
    def sample(self, batch_size):
        # 随机采样并转换为 Tensor
        return (state, action, next_state, reward, not_done)
```

#### 6.2 Tmux 并发管理
```python
# 查找可用窗口
find_window_and_execute_command(command)
├── 遍历 tmux 窗口索引
├── 检查窗口可用性
└── 发送训练命令

# 并发训练多个状态表示
for train in range(sample_count):
    for seed in range(train_seed_count):
        find_window_and_execute_command(
            f"CUDA_VISIBLE_DEVICES={cuda} python lesr_train.py ..."
        )
```

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                      LESR 数据流                                │
└─────────────────────────────────────────────────────────────────┘

环境原始状态
    ↓
revise_state(s) ←── LLM 生成的状态表示函数
    ↓
扩展状态 [original_state | computed_features]
    ↓
┌───────────────┐
│   Actor 网络  │ → 动作选择
└───────────────┘
    ↓
环境执行
    ↓
奖励 = 环境奖励 + intrinsic_w × intrinsic_reward(扩展状态)
    ↓
存储到 ReplayBuffer
    ↓
┌───────────────┐
│  Critic 网络  │ → Q 值学习
└───────────────┘
    ↓
Episode 结束
    ↓
Lipschitz 常数计算
    ↓
反馈给 LLM → 生成改进的状态表示
```

---

## 关键参数配置

### 迭代参数
- `--iteration`: 迭代轮数 (默认 5)
- `--sample_count`: 每轮采样数量 (默认 6)
- `--train_seed_count`: 每个样本训练种子数 (默认 1)
- `--evaluate_count`: 最终评估种子数 (默认 5)

### 训练参数
- `--max_timesteps`: 每次训练步数 (默认 800,000)
- `--max_evaluate_timesteps`: 最终评估步数 (默认 1,000,000)
- `--intrinsic_w`: 内在奖励权重 (默认 0.02)
- `--corr_tau`: 相关性软更新率 (默认 0.005)

### LLM 参数
- `--model`: 使用的模型 (默认 gpt-4-1106-preview)
- `--temperature`: 采样温度 (默认 0.0)
- `--openai_key`: OpenAI API 密钥

---

## 输出文件结构

```
LESR-resources/
├── run-v{version}-{env}/
│   ├── it_{iteration}_sample_{sample_id}.py     # LLM 生成的代码
│   ├── result/
│   │   ├── it{it}_train{train}_s{seed}.npy     # 训练结果
│   │   ├── it{it}_train{train}_corr_s{seed}.npy # Lipschitz 常数
│   │   └── evaluate_seed{seed}.npy              # 评估结果
│   ├── best_result/
│   │   └── v{version}-best-{env}.py             # 最佳状态表示
│   └── dialogs_it{it}.txt                       # LLM 对话记录
└── LESR-best-result/                            # 预训练最佳结果
    ├── HalfCheetah-v4.py
    ├── Walker2d-v4.py
    ├── Ant-v4.py
    └── ...
```

---

## 系统优势

1. **自动化特征工程**：LLM 自动设计任务相关特征，无需人工先验知识
2. **迭代优化**：通过反馈机制持续改进状态表示
3. **可解释性**：Lipschitz 常数分析提供状态-奖励关系的可解释性
4. **并行高效**：Tmux 并发训练加速迭代过程
5. **通用性强**：支持多种 MuJoCo 和 Robotics 任务

---

## 实验结果示例

根据 README 提供的数据：
- **MuJoCo 任务**：平均超过 baseline 29% 的累积奖励
- **Gym-Robotics 任务**：平均超过 baseline 30% 的成功率
- **泛化能力**：在新任务 (Walker Jump, Walker Split Legs) 上展现良好泛化性

---

## 技术栈

- **强化学习**：TD3 (Twin Delayed DDPG)
- **环境**：Gymnasium, MuJoCo, Gym-Robotics
- **LLM**：OpenAI GPT-4
- **并行**：Tmux, libtmux
- **数值计算**：NumPy, PyTorch
- **数据处理**：Pandas (读取 Excel 配置)
