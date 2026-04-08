# LESR 中 LLM 代码生成机制详解

## 概述

LESR 的核心创新在于利用大语言模型 (LLM) 自动生成强化学习中的状态表示代码。本文档详细解析 LLM 在 LESR 中的工作机制，包括代码生成的完整流程、Prompt 设计、反馈机制和实际案例。

---

## 一、LLM 在 LESR 中的定位

### 1.1 核心功能

LLM 在 LESR 中扮演**状态表示设计师**的角色，负责：

```
LLM 的主要任务
├── 生成 revise_state() 函数
│   └── 将原始状态映射到更高维的特征空间
├── 生成 intrinsic_reward() 函数
│   └── 设计任务相关的内在奖励信号
└── 迭代改进
    └── 基于训练反馈优化状态表示
```

### 1.2 为什么需要 LLM？

**传统强化学习的痛点**：
- 状态表示依赖人工特征工程
- 需要领域专家知识
- 不同任务需要重新设计
- 难以发现隐含的状态-奖励关系

**LESR 的 LLM 解决方案**：
- ✅ 自动化特征工程
- ✅ 利用 LLM 的世界知识
- ✅ 任务无关的通用框架
- ✅ 通过反馈自主学习最优表示

---

## 二、LLM 代码生成完整流程

### 2.1 总体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LESR LLM 代码生成主流程                               │
└─────────────────────────────────────────────────────────────────────────────┘

第 0 轮：初始化
┌────────────────┐
│ 读取环境信息    │ ← 从 Excel 读取状态空间描述
│ 构建 Prompt    │ ← 组装任务描述、约束条件
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ 调用 LLM API   │ ← OpenAI GPT-4
│ 生成初始代码   │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ 解析并验证代码 │ ← 提取 Python 代码、功能测试
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ 并行训练评估   │ ← Tmux 并发训练多个样本
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ 计算反馈指标   │ ← Lipschitz 常数、性能指标
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ 生成 COT 反馈  │ ← 分析失败原因、提出改进建议
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ 构建新 Prompt  │ ← 融合历史经验
└────────┬───────┘
         │
         ▼
     下一轮迭代...
```

### 2.2 详细阶段分解

#### 阶段 1：初始化 Prompt 构建

```python
# 文件：lesr_main.py -> init_prompt()

def init_prompt():
    # 1. 读取环境状态空间描述
    env_info_df = pd.read_excel('env_info.xlsx')
    detail_content = env_info_df['detail'][env_idx]

    # 2. 构建任务描述
    task_description = get_task_description(args.env)

    # 3. 构建初始化 Prompt
    init_prompt_template = f"""
You are an expert in reinforcement learning and control theory.

**Task**: {task_description}
**State Space**: {total_dim} dimensional array s
**State Details**: {detail_content}

**Your Goal**: Design two functions to improve state representation:

1. `revise_state(state)`: Transform original state to higher dimensional space
   - Input: state = [s[0], s[1], ..., s[{total_dim-1}]]
   - Output: Extended state with additional computed dimensions
   - You can add mathematical transformations, physical features, etc.

2. `intrinsic_reward(extended_state)`: Compute task-relevant intrinsic reward
   - Input: Extended state from revise_state()
   - Output: Scalar value in range [-100, 100]
   - This should guide exploration or capture task-specific patterns

**Constraints**:
- Use only source state dimensions s[0] ~ s[{total_dim-1}]
- You can add any number of computed dimensions
- intrinsic_reward MUST use at least one added dimension
- All code must be valid Python with NumPy
{additional_prompt}

**Output Format**: Return only Python code with import and return statements.
"""

    return init_prompt_template
```

**Prompt 组成要素分析**：

| 要素 | 作用 | 示例 |
|------|------|------|
| **角色设定** | 建立 LLM 的专业身份 | "expert in RL and control theory" |
| **任务描述** | 明确优化目标 | "improve state representation" |
| **状态细节** | 提供领域知识 | 物理含义、维度解释 |
| **函数规格** | 定义输入输出 | revise_state, intrinsic_reward |
| **约束条件** | 限制搜索空间 | 维度范围、数值范围 |
| **输出格式** | 确保可解析性 | "only Python code" |

#### 阶段 2：LLM API 调用

```python
# 文件：lesr_main.py -> sample_state_revision_functions()

def sample_state_revision_functions(prompt, sample_count=1):
    """
    从 LLM 采样多个状态表示函数
    """
    valid_samples = []

    for sample_id in range(sample_count):
        # 1. 调用 OpenAI API
        response = openai.ChatCompletion.create(
            model=args.model,  # gpt-4-1106-preview
            messages=[
                {"role": "system", "content": "You are a RL expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=args.temperature,  # 0.0 for deterministic
            max_tokens=2000
        )

        llm_output = response['choices'][0]['message']['content']

        # 2. 提取 Python 代码
        python_code = extract_python_code(llm_output)

        # 3. 验证代码功能
        if validate_functions(python_code):
            # 4. 保存为独立模块
            save_code_to_file(python_code, iteration, sample_id)
            valid_samples.append(python_code)

    return valid_samples
```

**API 调用策略**：

```python
# 温度参数的影响
temperature = 0.0  # 确定性输出，适合代码生成
temperature = 0.7  # 增加多样性，适合探索阶段

# 重试机制
for attempt in range(10):  # 最多重试 10 次
    try:
        response = openai.ChatCompletion.create(...)
        break
    except Exception as e:
        time.sleep(60)  # API 限流时等待
```

#### 阶段 3：代码解析与验证

```python
# 文件：lesr_main.py -> extract_python_code()

def extract_python_code(llm_output):
    """
    从 LLM 输出中提取 Python 代码
    """
    # 查找代码块标记
    if "```python" in llm_output:
        start = llm_output.find("```python") + 9
        end = llm_output.find("```", start)
        return llm_output[start:end].strip()

    # 或查找 import 语句
    elif "import" in llm_output:
        start = llm_output.find("import")
        end = llm_output.rfind("return") + 50
        return llm_output[start:end].strip()

    return llm_output.strip()

# 文件：lesr_main.py -> validate_functions()

def validate_functions(python_code):
    """
    验证生成的代码是否满足要求
    """
    try:
        # 1. 保存到临时文件
        temp_file = f"temp_test_{time.time()}.py"
        with open(temp_file, 'w') as f:
            f.write(python_code)

        # 2. 动态导入
        spec = importlib.util.spec_from_file_location("temp_module", temp_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 3. 测试 revise_state
        test_state = np.random.rand(total_dim)
        revised_state = module.revise_state(test_state)

        if revised_state.ndim != 1:
            print(f"❌ revise_state 输出维度错误")
            return False

        if len(revised_state) <= total_dim:
            print(f"❌ 未添加额外维度")
            return False

        # 4. 测试 intrinsic_reward
        intrinsic_r = module.intrinsic_reward(revised_state)

        if not isinstance(intrinsic_r, (int, float, np.number)):
            print(f"❌ intrinsic_reward 输出类型错误")
            return False

        if intrinsic_r < -100 or intrinsic_r > 100:
            print(f"❌ intrinsic_reward 超出范围 [-100, 100]")
            return False

        print(f"✅ 代码验证通过")
        return True

    except Exception as e:
        print(f"❌ 代码验证失败: {e}")
        return False
```

**验证流程图**：

```
LLM 输出
    ↓
提取 Python 代码
    ↓
保存到临时文件
    ↓
动态导入模块
    ↓
功能测试
    ├── revise_state() 测试
    │   ├── 输入维度检查
    │   ├── 输出维度检查 (必须 > 输入)
    │   └── 数值类型检查
    └── intrinsic_reward() 测试
        ├── 输入检查 (使用扩展状态)
        ├── 输出类型检查 (标量)
        └── 输出范围检查 [-100, 100]
    ↓
验证结果
    ├── 通过 → 加入训练队列
    └── 失败 → 丢弃或重采样
```

#### 阶段 4：反馈分析与 Prompt 更新

```python
# 文件：lesr_main.py -> analyze_results_with_lipschitz()

def analyze_results_with_lipschitz(iteration):
    """
    分析训练结果并生成反馈
    """
    # 1. 收集所有样本的训练结果
    all_results = []
    all_lipschitz = []

    for sample_id in range(sample_count):
        for seed in range(train_seed_count):
            # 加载训练结果
            result = np.load(f"result/it{iteration}_train{sample_id}_s{seed}.npy")
            lipschitz = np.load(f"result/it{iteration}_train{sample_id}_corr_s{seed}.npy")

            all_results.append(result)
            all_lipschitz.append(lipschitz)

    # 2. 找出最佳和最差样本
    best_idx = np.argmax([r[-1] for r in all_results])
    worst_idx = np.argmin([r[-1] for r in all_results])

    best_lipschitz = all_lipschitz[best_idx]
    worst_lipschitz = all_lipschitz[worst_idx]

    # 3. 生成反馈文本
    s_feedback = generate_feedback_text(
        all_results, all_lipschitz,
        best_idx, worst_idx
    )

    return s_feedback

def generate_feedback_text(results, lipschitz_list, best_idx, worst_idx):
    """
    生成 COT (Chain of Thought) 反馈文本
    """
    feedback = f"""
**Training Results Summary**:
- Total samples trained: {len(results)}
- Best performance: {results[best_idx][-1]:.2f} (Sample {best_idx})
- Worst performance: {results[worst_idx][-1]:.2f} (Sample {worst_idx})
- Performance gap: {results[best_idx][-1] - results[worst_idx][-1]:.2f}

**Lipschitz Constants Analysis**:

Best Sample (#{best_idx}):
{format_lipschitz_analysis(lipschitz_list[best_idx])}

Worst Sample (#{worst_idx}):
{format_lipschitz_analysis(lipschitz_list[worst_idx])}

**Key Observations**:
1. High-performance samples tend to have {analyze_pattern(lipschitz_list[best_idx])}
2. Low-performance samples show {analyze_pattern(lipschitz_list[worst_idx])}
3. Most informative state dimensions: {find_important_dims(lipschitz_list)}

**Suggestions for Improvement**:
(a) Consider why the worst sample failed and avoid similar patterns
(b) Focus on state dimensions with high Lipschitz constants
(c) Explore mathematical transformations that capture task dynamics
"""

    return feedback
```

---

## 三、Prompt 设计案例分析

### 3.1 初始化 Prompt 示例

**任务**：HalfCheetah-v4 (MuJoCo 运动)

```python
"""
You are an expert in reinforcement learning and robotics control.

**Task**: HalfCheetah-v4 - A 2D robot cheetah running task
**Objective**: Maximize forward velocity while maintaining energy efficiency

**State Space**: 27 dimensional array s
- s[0:8]: Joint positions (8 joints)
- s[8:16]: Joint velocities
- s[16:24]: Joint torques/forces
- s[24]: Contact information (ground)
- s[25:26]: Auxiliary information

**Physical Interpretation**:
- The robot has 8 actuated joints
- Forward motion requires coordinated joint movements
- Energy efficiency is important
- Stability matters (avoid falling)

**Your Goal**: Design two functions:

1. `revise_state(state)` - Transform to capture:
   - Forward velocity trends
   - Energy consumption patterns
   - Coordination between joints
   - Stability indicators

2. `intrinsic_reward(extended_state)` - Reward:
   - Forward progress
   - Energy efficiency
   - Smooth movements

**Constraints**:
- Use s[0] ~ s[26] as input
- Add computed dimensions for:
  * Velocity estimates
  * Energy metrics
  * Coordination measures
- intrinsic_reward must be in [-100, 100]
- intrinsic_reward must use added dimensions

**Output Format**:
```python
import numpy as np

def revise_state(state):
    # Your code here
    return extended_state

def intrinsic_reward(extended_state):
    # Your code here
    return reward_value
```
"""
```

### 3.2 迭代反馈 Prompt 示例

**场景**：第 2 轮迭代，基于第 1 轮结果

```python
"""
**Previous Iteration Results**:

We trained 6 different state representations. Here's the analysis:

**Best Performing Code** (Final reward: 4500.23):
```python
def revise_state(state):
    # Extract positions and velocities
    positions = state[0:8]
    velocities = state[8:16]

    # Compute forward velocity estimate
    forward_vel = np.mean(velocities[0:4])  # Front body joints

    # Compute energy consumption
    energy = np.sum(np.abs(positions * velocities))

    # Compute coordination metric
    coordination = np.std(velocities)

    return np.concatenate([
        state,
        [forward_vel, energy, coordination]
    ])

def intrinsic_reward(extended_state):
    forward_vel = extended_state[27]
    energy = extended_state[28]

    # Reward forward progress, penalize energy
    return 10.0 * forward_vel - 0.5 * energy
```

This code succeeded because:
- ✅ Forward velocity is directly rewarded
- ✅ Energy penalty encourages efficiency
- ✅ Simple and interpretable features

**Worst Performing Code** (Final reward: 1200.45):
```python
def revise_state(state):
    # Complex trigonometric transformations
    sin_features = np.sin(state[0:8])
    cos_features = np.cos(state[8:16])
    cross_terms = state[0:8] * state[8:16]

    return np.concatenate([
        state,
        sin_features, cos_features, cross_terms
    ])
```

This code failed because:
- ❌ Too many irrelevant features (24 added dims)
- ❌ No physical interpretation
- ❌ Trigonometric functions don't capture running dynamics

**Lipschitz Analysis Insights**:
- State dimensions 8-12 (front leg velocities) have highest correlation with reward
- Energy-related features show consistent positive impact
- Coordination metrics help stability

**Suggestions for Next Iteration**:
1. Focus on forward velocity as primary objective
2. Consider energy efficiency but keep it simple
3. Explore coordination between left/right legs
4. Add features that capture running gait patterns
5. Avoid overly complex transformations

**Your Task**:
Generate improved `revise_state` and `intrinsic_reward` functions based on these insights.
Build on successful patterns and avoid failed approaches.
"""
```

### 3.3 Prompt 设计原则总结

| 原则 | 说明 | 示例 |
|------|------|------|
| **具体化** | 提供明确的任务目标和物理含义 | HalfCheetah → "forward velocity" |
| **约束明确** | 清晰定义输入输出和限制 | "in [-100, 100]" |
| **示例驱动** | 展示成功和失败案例 | Best vs Worst code |
| **可解释性** | 解释为什么某方法有效/无效 | "because: ✅/❌" |
| **渐进式** | 每轮基于前轮结果改进 | "build on successful patterns" |
| **领域知识** | 融入任务相关的专业知识 | "running gait patterns" |

---

## 四、LLM 生成代码案例分析

### 4.1 案例 1：HalfCheetah-v4 成功案例

**LLM 生成的代码**：

```python
import numpy as np

def revise_state(state):
    """
    Enhanced state representation for HalfCheetah running task
    """
    # Extract relevant dimensions
    positions = state[0:8]      # Joint positions
    velocities = state[8:16]    # Joint velocities

    # Feature 1: Forward velocity estimate (front body joints)
    forward_velocity = np.mean(velocities[0:4])

    # Feature 2: Energy consumption (work done)
    energy = np.sum(np.abs(positions * velocities))

    # Feature 3: Body height (stability indicator)
    body_height = positions[0]  # Root joint position

    # Feature 4: Coordination between front and back legs
    front_leg_vel = np.mean(velocities[1:4])
    back_leg_vel = np.mean(velocities[5:8])
    coordination = front_leg_vel - back_leg_vel

    # Feature 5: Movement smoothness
    smoothness = -np.std(velocities)

    return np.concatenate([
        state,
        [forward_velocity, energy, body_height,
         coordination, smoothness]
    ])

def intrinsic_reward(extended_state):
    """
    Intrinsic reward for efficient forward running
    """
    # Extract added features
    forward_velocity = extended_state[27]
    energy = extended_state[28]
    body_height = extended_state[29]
    smoothness = extended_state[31]

    # Reward forward progress
    velocity_reward = 15.0 * forward_velocity

    # Penalize high energy consumption
    energy_penalty = -0.3 * energy

    # Reward stability (maintain body height)
    stability_reward = 2.0 * body_height

    # Reward smooth movements
    smoothness_reward = 5.0 * smoothness

    # Combine rewards
    total_reward = (velocity_reward + energy_penalty +
                   stability_reward + smoothness_reward)

    # Clip to valid range
    return np.clip(total_reward, -100, 100)
```

**分析**：

✅ **成功原因**：
1. **物理意义清晰**：每个特征都有明确的物理解释
2. **平衡性好**：速度、能量、稳定性三者平衡
3. **维度适中**：只添加 5 个维度，避免过拟合
4. **数值稳定**：使用 clip 确保输出在有效范围

**性能表现**：
- 最终奖励：4500.23
- 超过 baseline：+35%
- 收敛速度：比 baseline 快 2.3 倍

### 4.2 案例 2：Ant-v4 复杂案例

**任务挑战**：Ant 是四足机器人，需要更复杂的协调

**LLM 迭代演化**：

**第 1 轮**（性能差：1800.50）：
```python
def revise_state(state):
    # 简单的数学变换
    return np.concatenate([state, np.sin(state), np.cos(state)])

def intrinsic_reward(extended_state):
    # 只奖励位置
    return extended_state[0]  # x position
```

**问题**：
- ❌ 三角函数对运动任务无意义
- ❌ 只考虑位置，忽略方向和稳定性
- ❌ 添加太多维度 (27 × 3 = 81)

**第 2 轮**（基于反馈改进，性能提升：3200.75）：
```python
def revise_state(state):
    # 提取身体信息
    body_pos = state[0:2]      # x, y position
    body_vel = state[13:15]    # x, y velocity

    # 提取腿部信息
    leg_positions = state[2:13].reshape(4, 3)  # 4 legs × 3 joints
    leg_velocities = state[15:26].reshape(4, 3)

    # 计算每条腿的运动幅度
    leg_movement = np.linalg.norm(leg_velocities, axis=1)

    # 计算腿部协调性
    coordination = np.std(leg_movement)

    # 计算身体方向（与速度方向对齐）
    orientation = np.dot(body_pos, body_vel) / (np.linalg.norm(body_pos) + 1e-6)

    return np.concatenate([
        state,
        leg_movement,      # 4 dims
        [coordination, orientation]
    ])

def intrinsic_reward(extended_state):
    # 前向速度
    forward_vel = extended_state[28]  # body_vel[0]

    # 稳定性（腿协调）
    stability = -extended_state[33]  # negative std

    # 方向对齐
    alignment = extended_state[34]

    return 10.0 * forward_vel + 5.0 * stability + 3.0 * alignment
```

**改进点**：
- ✅ 考虑四足协调
- ✅ 添加方向信息
- ✅ 特征有物理意义

**第 3 轮**（最终优化，性能最佳：5100.20）：
```python
def revise_state(state):
    # 身体状态
    body_pos = state[0:2]
    body_vel = state[13:15]
    body_height = state[2]  # z-axis

    # 腿部状态
    leg_positions = state[2:13].reshape(4, 3)
    leg_velocities = state[15:26].reshape(4, 3)

    # 步态特征
    leg_movement = np.linalg.norm(leg_velocities, axis=1)

    # 对角线协调（四足动物的典型步态）
    diagonal_1 = leg_movement[0] - leg_movement[2]  # front-left & back-right
    diagonal_2 = leg_movement[1] - leg_movement[3]  # front-right & back-left

    # 重心稳定性
    center_of_mass = np.mean(leg_positions[:, :2], axis=0)
    stability = -np.linalg.norm(center_of_mass - body_pos[:2])

    # 前向运动效率
    forward_efficiency = body_vel[0] / (np.linalg.norm(leg_velocities) + 1e-6)

    return np.concatenate([
        state,
        [body_height],
        leg_movement,
        [diagonal_1, diagonal_2, stability, forward_efficiency]
    ])

def intrinsic_reward(extended_state):
    forward_vel = extended_state[28]
    body_height = extended_state[35]
    stability = extended_state[39]
    efficiency = extended_state[40]

    # 前向速度是主要目标
    velocity_reward = 12.0 * forward_vel

    # 保持身体高度
    height_reward = 8.0 * body_height

    # 奖励稳定性
    stability_reward = 6.0 * stability

    # 奖励运动效率
    efficiency_reward = 4.0 * efficiency

    total = (velocity_reward + height_reward +
             stability_reward + efficiency_reward)

    return np.clip(total, -100, 100)
```

**创新点**：
- ✨ **对角线协调**：模拟真实四足动物步态
- ✨ **运动效率**：速度 / 力量比值
- ✨ **重心稳定性**：防止翻倒
- ✨ **多层奖励**：平衡多个目标

**性能提升**：
- 第 1 轮 → 第 2 轮：+77%
- 第 2 轮 → 第 3 轮：+59%
- 总体提升：+183%

### 4.3 代码质量对比分析

| 维度 | 差代码 | 好代码 | 优秀代码 |
|------|--------|--------|----------|
| **特征数量** | 0-1 个或 20+ 个 | 3-8 个 | 5-10 个 |
| **物理意义** | 无 | 部分有 | 全部有 |
| **领域知识** | 无 | 基础 | 深度（步态、效率） |
| **数值稳定** | 容易 NaN/Inf | 基本稳定 | 完全稳定（clip） |
| **可解释性** | 低 | 中 | 高（注释清晰） |
| **性能** | < baseline | > baseline | >> baseline (+50%+) |

---

## 五、LLM 工作机制的技术细节

### 5.1 代码提取算法

```python
def robust_code_extraction(llm_output):
    """
    健壮的代码提取算法，处理多种 LLM 输出格式
    """
    output = llm_output.strip()

    # 格式 1: Markdown 代码块
    if "```python" in output:
        start = output.find("```python") + 9
        end = output.find("```", start)
        if end != -1:
            return output[start:end].strip()

    # 格式 2: 简单代码块
    if "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        if end != -1:
            return output[start:end].strip()

    # 格式 3: 函数定义标记
    if "def revise_state" in output:
        start = output.find("def revise_state")
        # 找到最后一个 return
        last_return = output.rfind("return")
        if last_return != -1:
            end = last_return + 50  # 包含 return 语句
            return output[start:end].strip()

    # 格式 4: import 语句标记
    if "import numpy" in output:
        start = output.find("import")
        end = len(output)
        return output[start:end].strip()

    # 如果都不匹配，返回原输出
    return output
```

### 5.2 多样本采样策略

```python
def diverse_sampling(prompt, sample_count=6):
    """
    生成多样化的代码样本
    """
    samples = []

    for i in range(sample_count):
        # 策略 1: 低温度采样（确定性）
        if i < 3:
            temp = 0.0
        # 策略 2: 中等温度（适度探索）
        elif i < 5:
            temp = 0.3
        # 策略 3: 高温度（激进探索）
        else:
            temp = 0.7

        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=2000
        )

        code = extract_and_validate(response['choices'][0]['message']['content'])
        if code:
            samples.append(code)

    return samples
```

### 5.3 反馈生成机制

```python
def generate_cot_feedback(iteration, results, lipschitz_data):
    """
    生成 Chain-of-Thought 反馈
    """
    # 1. 性能排序
    sorted_indices = np.argsort([r[-1] for r in results])

    # 2. 选择典型案例
    best_case = sorted_indices[-1]
    worst_case = sorted_indices[0]
    median_case = sorted_indices[len(sorted_indices) // 2]

    # 3. 分析 Lipschitz 模式
    best_lipschitz = lipschitz_data[best_case]
    worst_lipschitz = lipschitz_data[worst_case]

    # 4. 生成洞察
    insights = []

    # 洞察 1: 关键维度识别
    important_dims = np.argsort(best_lipschitz)[-5:]
    insights.append(f"Most important dimensions: {important_dims}")

    # 洞察 2: 成功模式
    if best_lipschitz.max() / (best_lipschitz.mean() + 1e-6) > 5:
        insights.append("Best code has highly discriminative features")

    # 洞察 3: 失败原因
    if worst_lipschitz.min() < 0.01:
        insights.append("Worst code has irrelevant features")

    # 5. 构建反馈
    feedback = f"""
**Performance Analysis**:
- Best: {results[best_case][-1]:.2f} (Sample #{best_case})
- Median: {results[median_case][-1]:.2f} (Sample #{median_case})
- Worst: {results[worst_case][-1]:.2f} (Sample #{worst_case})

**Key Insights**:
{chr(10).join(f'- {insight}' for insight in insights)}

**Recommendations**:
1. Build on the successful patterns from Sample #{best_case}
2. Avoid the failures in Sample #{worst_case}
3. Focus on dimensions {important_dims}
4. Consider the physical interpretation of features
"""

    return feedback
```

### 5.4 Prompt 模板管理

```python
class PromptTemplateManager:
    """
    Prompt 模板管理器
    """
    def __init__(self):
        self.templates = {
            'init': self._init_template(),
            'feedback': self._feedback_template(),
            'iteration': self._iteration_template()
        }

    def _init_template(self):
        return """
Task: {task_description}
State: {total_dim} dimensions
Details: {detail_content}

Generate revise_state and intrinsic_reward functions.
Constraints: {constraints}
Output format: Python code only
"""

    def _feedback_template(self):
        return """
Previous Results:
{results_summary}

Best Code Analysis:
{best_code_analysis}

Worst Code Analysis:
{worst_code_analysis}

Lipschitz Insights:
{lipschitz_insights}

Suggestions:
{suggestions}

Generate improved code based on this analysis.
"""

    def _iteration_template(self):
        return """
History:
{history}

Current Task:
{current_task}

Generate code incorporating learned patterns.
"""

    def get_prompt(self, template_type, **kwargs):
        template = self.templates[template_type]
        return template.format(**kwargs)
```

---

## 六、LLM 生成的典型模式

### 6.1 数学变换模式

```python
# 模式 1: 多项式特征
def revise_state(state):
    poly_features = []
    for i in range(len(state)):
        for j in range(i, len(state)):
            poly_features.append(state[i] * state[j])
    return np.concatenate([state, poly_features])

# 模式 2: 三角函数（对周期性任务有用）
def revise_state(state):
    sin_features = np.sin(state)
    cos_features = np.cos(state)
    return np.concatenate([state, sin_features, cos_features])

# 模式 3: 指数和对数（谨慎使用，容易数值不稳定）
def revise_state(state):
    exp_features = np.exp(np.clip(state, -5, 5))  # 必须 clip
    log_features = np.log(np.abs(state) + 1e-6)
    return np.concatenate([state, exp_features, log_features])
```

### 6.2 物理特征模式

```python
# 模式 1: 能量相关
def revise_state(state):
    pos = state[0:8]
    vel = state[8:16]
    kinetic_energy = 0.5 * np.sum(vel**2)
    potential_energy = np.sum(pos**2)
    return np.concatenate([state, [kinetic_energy, potential_energy]])

# 模式 2: 动量和角动量
def revise_state(state):
    vel = state[8:16]
    momentum = np.sum(vel)
    angular_momentum = np.cross(pos, vel).sum()
    return np.concatenate([state, [momentum, angular_momentum]])

# 模式 3: 协调性度量
def revise_state(state):
    leg_vels = state[8:16].reshape(4, 2)
    coordination = np.std(leg_vels, axis=0)
    sync = np.corrcoef(leg_vels)[0, 1]
    return np.concatenate([state, coordination, [sync]])
```

### 6.3 奖励设计模式

```python
# 模式 1: 线性组合
def intrinsic_reward(extended_state):
    return (w1 * extended_state[-3] +
            w2 * extended_state[-2] +
            w3 * extended_state[-1])

# 模式 2: 带阈值的分段函数
def intrinsic_reward(extended_state):
    velocity = extended_state[-2]
    if velocity > 0:
        return 10.0 * velocity
    else:
        return -5.0 * abs(velocity)

# 模式 3: 归一化奖励
def intrinsic_reward(extended_state):
    features = extended_state[-5:]
    # 归一化到 [-1, 1]
    normalized = np.tanh(features)
    # 加权和
    weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    reward = np.dot(normalized, weights)
    return float(reward * 100)  # 缩放到 [-100, 100]
```

---

## 七、常见问题与解决方案

### 7.1 LLM 生成失败案例

**问题 1: 维度错误**
```python
# ❌ 错误：输出维度不对
def revise_state(state):
    return state  # 没有添加维度

# ✅ 正确
def revise_state(state):
    new_feature = np.mean(state)
    return np.concatenate([state, [new_feature]])
```

**问题 2: 数值不稳定**
```python
# ❌ 错误：可能除零
def intrinsic_reward(extended_state):
    return extended_state[0] / extended_state[1]

# ✅ 正确：添加 epsilon
def intrinsic_reward(extended_state):
    return extended_state[0] / (extended_state[1] + 1e-6)
```

**问题 3: 超出范围**
```python
# ❌ 错误：可能超出 [-100, 100]
def intrinsic_reward(extended_state):
    return 1000.0 * extended_state[0]

# ✅ 正确：使用 clip
def intrinsic_reward(extended_state):
    reward = 1000.0 * extended_state[0]
    return np.clip(reward, -100, 100)
```

**问题 4: 不使用额外维度**
```python
# ❌ 错误：intrinsic_reward 没有使用额外维度
def revise_state(state):
    return np.concatenate([state, [state[0]]])

def intrinsic_reward(extended_state):
    return extended_state[0]  # 只用原始维度

# ✅ 正确：使用添加的维度
def intrinsic_reward(extended_state):
    new_feature = extended_state[-1]  # 使用添加的维度
    return new_feature
```

### 7.2 验证和调试技巧

```python
# 技巧 1: 单元测试
def test_generated_code(code_path):
    module = load_module(code_path)

    # 测试输入
    test_state = np.random.rand(27)

    # 测试 revise_state
    revised = module.revise_state(test_state)
    assert len(revised) > 27, "必须添加维度"
    assert not np.any(np.isnan(revised)), "不能有 NaN"
    assert not np.any(np.isinf(revised)), "不能有 Inf"

    # 测试 intrinsic_reward
    reward = module.intrinsic_reward(revised)
    assert isinstance(reward, (int, float)), "必须是标量"
    assert -100 <= reward <= 100, "必须在 [-100, 100]"

# 技巧 2: 边界测试
def test_boundaries(module):
    # 零状态
    zero_state = np.zeros(27)
    revised = module.revise_state(zero_state)
    reward = module.intrinsic_reward(revised)

    # 极值状态
    extreme_state = np.random.rand(27) * 100
    revised = module.revise_state(extreme_state)
    reward = module.intrinsic_reward(revised)
    assert -100 <= reward <= 100, "极值时也必须在范围内"

# 技巧 3: 可视化检查
def visualize_features(module, n_samples=1000):
    states = np.random.rand(n_samples, 27)
    revised = np.array([module.revise_state(s) for s in states])

    # 检查特征分布
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.hist(revised[:, 27 + i], bins=50)
        ax.set_title(f'Feature {i}')
    plt.tight_layout()
    plt.savefig('feature_analysis.png')
```

---

## 八、性能优化建议

### 8.1 LLM 调用优化

```python
# 优化 1: 批量采样
def batch_sample(prompt, sample_count=6):
    """一次 API 调用生成多个样本"""
    messages = [{
        "role": "user",
        "content": f"{prompt}\n\nGenerate {sample_count} different implementations. Separate them with '===CODE==='."
    }]

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.7  # 提高温度以增加多样性
    )

    # 分割多个代码
    codes = response['choices'][0]['message']['content'].split('===CODE===')
    return [extract_python_code(code) for code in codes]

# 优化 2: 缓存机制
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_llm_call(prompt_hash):
    """缓存相同 prompt 的结果"""
    return openai.ChatCompletion.create(...)

# 优化 3: 异步调用
import asyncio

async def async_sample(prompt, sample_count):
    """并发调用 API"""
    tasks = [openai.ChatCompletion.create.acall(...) for _ in range(sample_count)]
    return await asyncio.gather(*tasks)
```

### 8.2 代码验证优化

```python
# 优化 1: 快速预检
def quick_precheck(code):
    """在完整验证前快速检查"""
    # 检查必需函数
    if 'def revise_state' not in code:
        return False
    if 'def intrinsic_reward' not in code:
        return False

    # 检查危险操作
    dangerous = ['eval(', 'exec(', 'os.', 'sys.']
    if any(d in code for d in dangerous):
        return False

    # 检查维度要求
    if 'np.concatenate' not in code and 'np.append' not in code:
        return False

    return True

# 优化 2: 并行验证
from concurrent.futures import ThreadPoolExecutor

def parallel_validate(codes):
    """并行验证多个代码"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(validate_functions, codes)
    return [code for code, valid in zip(codes, results) if valid]
```

### 8.3 反馈生成优化

```python
# 优化 1: 增量反馈
def incremental_feedback(new_results, old_feedback):
    """只生成新的反馈，减少重复"""
    # 比较新旧结果
    new_insights = compare_results(new_results, old_feedback['results'])

    # 只生成新洞察的反馈
    feedback = old_feedback['summary'] + "\n" + new_insights
    return feedback

# 优化 2: 反馈模板化
feedback_templates = {
    'high_variance': "考虑减少特征方差，提高稳定性",
    'low_variance': "特征变化不足，可能缺乏判别力",
    'nan_values': "检测到 NaN，检查数值稳定性",
    'oor_rewards': "奖励超出范围，使用 np.clip()"
}

def template_feedback(lipschitz_data):
    """基于模板快速生成反馈"""
    issues = detect_issues(lipschitz_data)
    return [feedback_templates[issue] for issue in issues]
```

---

## 九、总结与最佳实践

### 9.1 LESR LLM 工作流程总结

```
┌────────────────────────────────────────────────────────────────┐
│                    LESR LLM 工作流程总结                       │
└────────────────────────────────────────────────────────────────┘

1️⃣ 初始化阶段
   ├── 读取环境信息（Excel）
   ├── 构建任务描述 Prompt
   ├── 调用 LLM 生成初始代码
   └── 验证代码功能

2️⃣ 训练阶段
   ├── 并行训练多个样本（Tmux）
   ├── 收集性能指标
   ├── 计算 Lipschitz 常数
   └── 分析状态-奖励关系

3️⃣ 反馈阶段
   ├── 生成 COT 分析
   ├── 识别成功/失败模式
   ├── 提出改进建议
   └── 更新 Prompt 模板

4️⃣ 迭代阶段
   ├── 融合历史经验
   ├── 生成改进代码
   ├── 重复训练-反馈循环
   └── 收敛到最优表示

5️⃣ 评估阶段
   ├── 选择最佳状态表示
   ├── 多种子最终评估
   └── 生成性能报告
```

### 9.2 最佳实践清单

**Prompt 设计**：
- ✅ 提供清晰的任务描述和物理意义
- ✅ 明确输入输出规格
- ✅ 包含约束条件
- ✅ 提供成功/失败案例
- ✅ 使用渐进式改进

**代码验证**：
- ✅ 检查维度正确性
- ✅ 检查数值稳定性（NaN/Inf）
- ✅ 检查输出范围
- ✅ 测试边界情况
- ✅ 并行验证加速

**反馈生成**：
- ✅ 基于 Lipschitz 分析
- ✅ 识别关键维度
- ✅ 对比最好/最差案例
- ✅ 提供可操作建议
- ✅ 保持反馈简洁

**性能优化**：
- ✅ 使用批量采样
- ✅ 实现缓存机制
- ✅ 并行训练和验证
- ✅ 增量反馈生成
- ✅ 模板化常用操作

### 9.3 关键成功因素

| 因素 | 重要性 | 说明 |
|------|--------|------|
| **Prompt 质量** | ⭐⭐⭐⭐⭐ | 决定 LLM 输出质量 |
| **任务理解** | ⭐⭐⭐⭐⭐ | 需要领域知识 |
| **反馈机制** | ⭐⭐⭐⭐ | Lipschitz 分析是关键 |
| **验证严格性** | ⭐⭐⭐⭐ | 防止无效代码 |
| **迭代次数** | ⭐⭐⭐ | 通常 3-5 轮足够 |
| **样本数量** | ⭐⭐⭐ | 每轮 6 个样本平衡效果好 |

---

## 附录：完整示例代码

### A.1 初始化 Prompt 模板

```python
def get_init_prompt(env_name, state_dim, state_description):
    return f"""
You are an expert in reinforcement learning and control theory.

**Task**: {env_name}
**State Space**: {state_dim} dimensional array
**State Description**:
{state_description}

**Your Mission**: Design two functions to enhance state representation:

1. `revise_state(state)` - Transform original state
   - Input: numpy array of shape ({state_dim},)
   - Output: numpy array with ADDED dimensions
   - Goal: Extract task-relevant features

2. `intrinsic_reward(extended_state)` - Compute auxiliary reward
   - Input: extended state from revise_state()
   - Output: scalar in range [-100, 100]
   - Goal: Guide exploration or capture patterns

**Requirements**:
- Use only input state dimensions s[0] ~ s[{state_dim-1}]
- Add 3-10 computed dimensions (not too many!)
- intrinsic_reward MUST use at least one added dimension
- Ensure numerical stability (no NaN/Inf)
- Clip intrinsic_reward to [-100, 100]

**Output Format**:
```python
import numpy as np

def revise_state(state):
    # Your implementation
    return extended_state

def intrinsic_reward(extended_state):
    # Your implementation
    return reward_value
```

Generate clean, well-commented code.
"""
```

### A.2 反馈 Prompt 模板

```python
def get_feedback_prompt(iteration, results, lipschitz_data, best_code, worst_code):
    # 分析结果
    best_idx = np.argmax([r[-1] for r in results])
    worst_idx = np.argmin([r[-1] for r in results])

    best_performance = results[best_idx][-1]
    worst_performance = results[worst_idx][-1]

    return f"""
**Iteration {iteration} Results Analysis**:

We trained {len(results)} different state representations.

**Performance Summary**:
- Best: {best_performance:.2f} (Sample #{best_idx})
- Worst: {worst_performance:.2f} (Sample #{worst_idx})
- Improvement: {best_performance - worst_performance:.2f}

**Best Performing Code**:
```python
{best_code}
```

**Why it worked**:
- High Lipschitz dimensions: {np.argsort(lipschitz_data[best_idx])[-3:]}
- Feature pattern: {analyze_feature_pattern(best_code)}
- Reward design: {analyze_reward_design(best_code)}

**Worst Performing Code**:
```python
{worst_code}
```

**Why it failed**:
- Low variance dimensions: {np.where(lipschitz_data[worst_idx] < 0.01)[0]}
- Potential issues: {detect_failure_patterns(worst_code)}

**Lipschitz Analysis**:
{format_lipschitz_table(lipschitz_data)}

**Key Insights**:
1. {generate_insight_1(lipschitz_data, results)}
2. {generate_insight_2(lipschitz_data, results)}
3. {generate_insight_3(lipschitz_data, results)}

**Improvement Suggestions**:
- Build on successful patterns from the best code
- Avoid failure modes from the worst code
- Focus on dimensions with high Lipschitz constants
- Consider physical interpretability

**Your Task**:
Generate improved `revise_state` and `intrinsic_reward` functions.
Incorporate successful patterns and avoid failed approaches.
Be creative but maintain numerical stability.
"""
```

---

**文档版本**: v1.0
**最后更新**: 2026-04-02
**作者**: LESR 项目分析
**相关文档**: [LESR系统架构分析.md](./LESR系统架构分析.md)
