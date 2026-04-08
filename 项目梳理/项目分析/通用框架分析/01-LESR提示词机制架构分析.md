# LESR 提示词机制架构分析：通用框架与任务定制

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LESR LLM 提示词系统架构                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         通用框架层 (Framework Layer)                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                  │
│  │ Prompt 模板引擎 │  │ 代码验证框架   │  │ 反馈分析框架   │                  │
│  └────────────────┘  └────────────────┘  └────────────────┘                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                      任务定制层 (Task Customization Layer)                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                  │
│  │ 状态空间描述   │  │ 任务特定约束   │  │ 领域知识注入   │                  │
│  │ (Excel配置)    │  │ (Additional)   │  │ (Domain Tips)  │                  │
│  └────────────────┘  └────────────────┘  └────────────────┘                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                        动态迭代层 (Dynamic Iteration Layer)                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                  │
│  │ 初始化 Prompt  │→│ COT 反馈 Prompt│→│ 迭代改进 Prompt │                  │
│  └────────────────┘  └────────────────┘  └────────────────┘                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 二、通用框架 vs 任务定制

### 2.1 通用框架部分（可迁移的核心）

```python
# ========== 通用组件 1: 初始化 Prompt 模板骨架 ==========
"""
Revise the state representation for a reinforcement learning agent.
=========================================================
The agent's task description is:
{task_description}                    # ← 任务定制占位符
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}                      # ← 任务定制占位符

You should design a task-related state representation based on the the source {total_dim} dim
to better for reinforcement training, using the detailed information mentioned above
to do some caculations, and feel free to do complex caculations, and then concat them to the source state.

Besides, we want you to design an intrinsic reward function based on the revise_state python function.

That is to say, we will:
1. use your revise_state python function to get an updated state: updated_s = revise_state(s)
2. use your intrinsic reward function to get an intrinsic reward for the task: r = intrinsic_reward(updated_s)
3. to better design the intrinsic_reward, we recommond you use some source dim in the updated_s,
   which is between updated_s[0] and updated_s[{total_dim - 1}]
4. however, you must use the extra dim in your given revise_state python function,
   which is between updated_s[{total_dim}] and the end of updated_s

{additional_prompt}                    # ← 任务定制占位符

Your task is to create two Python functions, named `revise_state`, which takes the current state `s`
as input and returns an updated state representation, and named `intrinsic_reward`, which takes the
updated state `updated_s` as input and returns an intrinsic reward.

The goal is to better for reinforcement training. Lets think step by step.
"""

# ========== 通用组件 2: COT 反馈 Prompt 模板骨架 ==========
"""
We have successfully trained Reinforcement Learning (RL) policy using {sample_count} different
state revision codes and intrinsic reward function codes sampled by you, and each pair of code
is associated with the training of a policy relatively.

Throughout every state revision code's training process, we monitored:
1. The final policy performance(accumulated reward).
2. Most importantly, every state revise dim's Lipschitz constant with the reward. That is to say,
   you can see which state revise dim is more related to the reward and which dim can contribute
   to enhancing the continuity of the reward function mapping. Lower Lipchitz constant means
   better continuity and smoother of the mapping. Note: Lower Lipchitz constant is better.

Here are the results:
{training_results}                      # ← 动态生成

You should analyze the results mentioned above and give suggestions about how to imporve the
performace of the "state revision code".

Here are some tips for how to analyze the results:
(a) if you find a state revision code's performance is very low, then you should analyze to figure out why it fail
(b) if you find some dims' are more related to the final performance, you should analyze to figure out what makes it success
(c) you should also analyze how to imporve the performace of the "state revision code" and "intrinsic reward code" later

Lets think step by step. Your solution should aim to improve the overall performance of the RL policy.
"""

# ========== 通用组件 3: 代码验证框架 ==========
def validate_functions(python_code, test_state):
    """
    通用验证逻辑（任务无关）:
    1. 代码可执行性检查
    2. 维度约束检查（输出 > 输入）
    3. 数值稳定性检查（无 NaN/Inf）
    4. 输出范围检查（intrinsic_reward ∈ [-100, 100]）
    """
    # ... 通用验证代码 ...

# ========== 通用组件 4: 反馈分析框架 ==========
def analyze_results_with_lipschitz(iteration, results, lipschitz_data):
    """
    通用反馈分析逻辑（任务无关）:
    1. 性能排序
    2. 最好/最差样本识别
    3. Lipschitz 常数分析
    4. 关键维度识别
    """
    # ... 通用分析代码 ...
```

### 2.2 任务定制部分（需要针对具体任务配置）

```python
# ========== 任务定制 1: 状态空间语义描述 (Excel 配置) ==========
# 文件: mujoco_observation_space.xlsx
# Sheet: halfcheetah

"""
Row 0: [0, 'z-coordinate of the front tip', -inf, inf, 'rootz', 'slide', 'position (m)', task_description]
Row 1: [1, 'angle of the front tip', -inf, inf, 'rooty', 'hinge', 'angle (rad)', nan]
Row 2: [2, 'angle of the second rotor', -inf, inf, 'bthigh', 'hinge', 'angle (rad)', nan]
...
Row 8: [8, 'x-coordinate of the front tip', -inf, inf, 'rootx', 'slide', 'velocity (m/s)', nan]
"""

# 解析后的 detail_content 格式:
"""
- `s[0]`: z-coordinate of the front tip , the unit is position (m).
- `s[1]`: angle of the front tip , the unit is angle (rad).
- `s[2]`: angle of the second rotor , the unit is angle (rad).
...
"""

# ========== 任务定制 2: 任务描述 (Task Description) ==========
"""
The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them
(including two paws). The goal is to apply a torque on the joints to make the cheetah run forward
(right) as fast as possible, with a positive reward allocated based on the distance moved forward
and a negative reward allocated for moving backward. The torso and head of the cheetah are fixed,
and the torque can only be applied on the other 6 joints over the front and back thighs (connecting
to the torso), shins (connecting to the thighs) and feet (connecting to the shins).
"""

# ========== 任务定制 3: 领域特定提示 (Additional Prompt) ==========
# AntMaze/Fetch/Adroit 任务需要身体协调能力

if 'antmaze' in args.env.lower() or 'fetch' in args.env.lower() or 'adroit' in args.env.lower():
    additional_prompt = """
Most Importantly: As for this task, the agent should firstly learn how to coordinate various
parts of the body itself before it finally reach the goal. Like a baby should learn how to walk
before finally walking to the goal. Therefore, when you design state representation and reward,
you should cosider how to make the agent learn to coordinate various parts of the body as well
as how to finally reach the goal.
"""
else:
    additional_prompt = ''  # HalfCheetah 等任务不需要额外提示

# ========== 任务定制 4: 性能指标语义 ==========
# 根据任务类型动态调整性能指标描述

if 'antmaze' in args.env.lower() or 'fetch' in args.env.lower() or 'adroit' in args.env.lower():
    policy_performance = 'Final Policy Success Rate'  # 迷宫/抓取任务关注成功率
else:
    policy_performance = 'Final Policy Performance'   # 运动任务关注累积奖励
```

### 2.3 通用框架与任务定制的分界线

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         分界线：什么是通用 vs 什么是定制                        │
└─────────────────────────────────────────────────────────────────────────────┘

【通用框架 - 跨任务迁移的组件】
✅ Prompt 模板结构（占位符系统）
✅ 代码验证逻辑（维度检查、数值稳定性）
✅ 反馈分析框架（Lipschitz 分析、性能排序）
✅ 迭代优化循环（COT 反馈 → 改进）
✅ 输出格式规范（Python 函数签名）

【任务定制 - 需要针对具体任务配置】
⚙️ 状态空间语义描述（Excel 配置文件）
⚙️ 任务目标描述（Task Description）
⚙️ 领域特定提示（Additional Prompt）
⚙️ 性能指标语义（Success Rate vs Cumulative Reward）
⚙️ 物理约束和单位（m, rad, m/s, rad/s）
```

## 三、提示词设计的核心理念

### 3.1 设计原则图示

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LESR 提示词设计三角原则                                │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌───────────────────┐
                        │   语义丰富性       │
                        │ (Semantic Richness)│
                        │                   │
                        │ - 物理含义         │
                        │ - 单位信息         │
                        │ - 任务目标         │
                        └─────────┬─────────┘
                                  │
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
        ┌───────┴───────┐         │         ┌───────┴───────┐
        │  结构化约束   │─────────┼─────────│  可执行验证   │
        │ (Structured   │         │         │ (Executable   │
        │  Constraints) │         │         │  Validation)  │
        │               │         │         │               │
        │ - 函数签名    │         │         │ - 维度检查    │
        │ - 输入输出    │         │         │ - 数值稳定性  │
        │ - 范围限制    │         │         │ - 代码可运行  │
        └───────────────┘         │         └───────────────┘
                                  │
                        ┌─────────┴─────────┐
                        │   迭代反馈闭环     │
                        │ (Iterative Feedback)│
                        │                   │
                        │ - COT 分析        │
                        │ - Lipschitz 洞察  │
                        │ - 成功/失败案例   │
                        └───────────────────┘
```

### 3.2 为什么要这样设计？

| 设计原则 | 理由 | 示例 |
|---------|------|------|
| **语义丰富性** | LLM 需要理解状态的物理含义才能设计有意义的特征 | HalfCheetah: "s[8] 是 x 方向速度，应该奖励向前速度" |
| **结构化约束** | 确保生成的代码可直接集成到 RL 流程 | `def revise_state(s): return updated_s` 固定签名 |
| **可执行验证** | 过滤无效代码，避免浪费训练资源 | 维度检查：`len(updated_s) > len(s)` |
| **迭代反馈** | LLM 作为优化器需要知道什么有效、什么无效 | Lipschitz 分析揭示哪些维度与奖励相关 |
| **领域知识注入** | 利用 LLM 的世界知识加速探索 | AntMaze: "像婴儿先学走路再学走向目标" |

## 四、三种 Prompt 的详细分析

### 4.1 初始化 Prompt (Init Prompt)

```python
# ========== 核心结构 ==========
init_prompt = f"""
Revise the state representation for a reinforcement learning agent.
=========================================================
The agent's task description is:
{task_description}                         # ← 任务定制：目标
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}                           # ← 任务定制：状态语义
- `s[0]`: z-coordinate of the front tip, the unit is position (m).
- `s[1]`: angle of the front tip, the unit is angle (rad).
...

You should design a task-related state representation based on the the source {total_dim} dim
to better for reinforcement training, using the detailed information mentioned above
to do some caculations, and feel free to do complex caculations, and then concat them to the source state.

Besides, we want you to design an intrinsic reward function based on the revise_state python function.

That is to say, we will:
1. use your revise_state python function to get an updated state: updated_s = revise_state(s)
2. use your intrinsic reward function to get an intrinsic reward for the task: r = intrinsic_reward(updated_s)
3. to better design the intrinsic_reward, we recommond you use some source dim in the updated_s,
   which is between updated_s[0] and updated_s[{total_dim - 1}]
4. however, you must use the extra dim in your given revise_state python function,
   which is between updated_s[{total_dim}] and the end of updated_s

{additional_prompt}                         # ← 任务定制：领域提示

Your task is to create two Python functions, named `revise_state`, which takes the current state `s`
as input and returns an updated state representation, and named `intrinsic_reward`, which takes the
updated state `updated_s` as input and returns an intrinsic reward. The functions should be
executable and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative
example of the expected output:

```python
import numpy as np
def revise_state(s):
    # Your state revision implementation goes here
    return updated_s
def intrinsic_reward(updated_s):
    # Your intrinsic reward code implementation goes here
    return intrinsic_reward
```
"""
```

**设计要点**：
1. **任务描述在前**：让 LLM 先理解"要做什么"
2. **状态语义详尽**：每个维度的物理含义 + 单位
3. **约束清晰**：明确输入输出规格
4. **示例引导**：提供代码模板，降低格式错误率

### 4.2 COT 反馈 Prompt (Chain-of-Thought Feedback)

```python
# ========== 核心结构 ==========
cot_prompt = f"""
We have successfully trained Reinforcement Learning (RL) policy using {args.sample_count} different
state revision codes and intrinsic reward function codes sampled by you, and each pair of code
is associated with the training of a policy relatively.

Throughout every state revision code's training process, we monitored:
1. The final policy performance(accumulated reward).
2. Most importantly, every state revise dim's Lipschitz constant with the reward. That is to say,
   you can see which state revise dim is more related to the reward and which dim can contribute
   to enhancing the continuity of the reward function mapping. Lower Lipchitz constant means
   better continuity and smoother of the mapping. Note: Lower Lipchitz constant is better.

Here are the results:
{s_feedback}                               # ← 动态生成：每个样本的代码 + 性能 + Lipschitz

========== State Revise and Intrinsic Reward Code -- 1 ==========
{code_sample_1}
========== State Revise and Intrinsic Reward Code -- 1's Final Policy Performance: 4500.23 ==========
In this State Revise Code 1, the source state dim is from s[0] to s[26], the Lipchitz constant
between them and the reward are(Note: The Lipschitz constant is always greater than or equal to 0,
and a lower Lipschitz constant implies better smoothness.):
s[0] lipschitz constant with reward = 0.02
s[8] lipschitz constant with reward = 0.85  # ← 高相关性：前向速度
...
In this State Revise Code 1, you give 5 extra dim from s[27] to s[31], the lipschitz constant
between them and the reward are:
s[27] lipschitz constant with reward = 0.92  # ← 添加的前向速度特征非常有效
...
======================================================================

You should analyze the results mentioned above and give suggestions about how to imporve the
performace of the "state revision code".

Here are some tips for how to analyze the results:
(a) if you find a state revision code's performance is very low, then you should analyze to figure out why it fail
(b) if you find some dims' are more related to the final performance, you should analyze to figure out what makes it success
(c) you should also analyze how to imporve the performace of the "state revision code" and "intrinsic reward code" later

Lets think step by step. Your solution should aim to improve the overall performance of the RL policy.
"""
```

**设计要点**：
1. **定量数据驱动**：Lipschitz 常数提供可解释的反馈
2. **对比分析**：好样本 vs 差样本的代码对比
3. **分析引导**：(a)(b)(c) 提示如何思考
4. **聚焦关键维度**：指出哪些维度最重要

### 4.3 迭代改进 Prompt (Next Iteration Prompt)

```python
# ========== 核心结构 ==========
next_iteration_prompt = f"""
Revise the state representation for a reinforcement learning agent.
=========================================================
The agent's task description is:
{task_description}
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}

You should design a task-related state representation based on the source {total_dim} dim
to better for reinforcement training, using the detailed information mentioned above
to do some caculations, and feel free to do complex caculations, and then concat them to the source state.

# ========== 关键：历史经验 ==========
For this problem, we have some history experience for you, here are some state revision
codes we have tried in the former iterations:
{former_histoy}

Former Iteration:1
========== State Revise and Intrinsic Reward Code -- 1 ==========
{code_sample_1}
========== Final Policy Performance: 4500.23 ==========
========== Lipschitz Analysis ==========
s[27] (forward_velocity): 0.92 ← 高相关性
...
======================================================================

From Former Iteration:1, we have some suggestions for you:
{cot_suggestions_1}
"The best code succeeded because it focused on forward velocity as the primary feature.
The worst code failed because it added too many irrelevant trigonometric features.
Consider: (1) Focus on forward velocity, (2) Avoid overly complex transformations..."
======================================================================

# ========== 基于历史改进 ==========
Based on the former suggestions. We are seeking an improved state revision code and an
improved intrinsic reward code that can enhance the model's performance on the task.
The state revised code should incorporate calculations, and the results should be
concatenated to the original state.

Besides, We are seeking an improved intrinsic reward code.

That is to say, we will:
1. use your revise_state python function to get an updated state: updated_s = revise_state(s)
2. use your intrinsic reward function to get an intrinsic reward for the task: r = intrinsic_reward(updated_s)
3. to better design the intrinsic_reward, we recommond you use some source dim in the updated_s,
   which is between updated_s[0] and updated_s[{total_dim - 1}]
4. however, you must use the extra dim in your given revise_state python function,
   which is between updated_s[{total_dim}] and the end of updated_s
{additional_prompt}

Your task is to create two Python functions, named `revise_state`, which takes the current state `s`
as input and returns an updated state representation, and named `intrinsic_reward`, which takes the
updated state `updated_s` as input and returns an intrinsic reward. The functions should be
executable and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step.
"""
```

**设计要点**：
1. **历史上下文**：展示所有迭代的历史
2. **经验传承**：COT 建议直接嵌入
3. **渐进式改进**：基于成功模式，避免失败模式
4. **保持约束**：每轮都重申核心约束

## 五、任务定制的提示词撰写思路

### 5.1 撰写流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    任务定制 Prompt 撰写七步法                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: 任务抽象理解
├─ 目标是什么？ (目标导向 vs 探索导向)
├─ 状态空间语义？ (位置/速度/力矩/接触)
└─ 关键物理量？ (前向速度/能量/稳定性)

Step 2: 状态空间语义标注
├─ 每个 s[i] 的物理含义
├─ 单位 (m/rad/m/s/rad/s)
├─ 取值范围 (如有)
└─ 关键维度识别 (哪些最相关)

Step 3: 任务描述撰写
├─ 主体是什么？ (机器人/智能体)
├─ 环境特征？ (2D/3D, 地形, 障碍)
├─ 目标是什么？ (前向运动/到达目标/抓取)
└─ 奖励机制？ (距离/速度/成功率)

Step 4: 领域知识注入
├─ 需要哪些先验？ (身体协调/步态模式)
├─ 常见失败模式？ (摔倒/原地转圈)
└─ 关键成功因素？ (平衡/效率/方向)

Step 5: 约束条件设计
├─ 输入输出规格 (维度范围)
├─ 数值稳定性 (clip/epsilon)
└─ 物理合理性 (能量守恒)

Step 6: 示例代码模板
├─ 函数签名固定
├─ 简单注释引导
└─ 返回值明确

Step 7: 迭代反馈设计
├─ 性能指标 (Success Rate vs Reward)
├─ 分析维度 (Lipschitz/特征重要性)
└─ 改进建议 (成功模式/失败模式)
```

### 5.2 实例：HalfCheetah 任务定制

```python
# ========== Step 1: 任务抽象理解 ==========
"""
任务类型: 目标导向 (最大化前向速度)
关键物理量:
  - 前向速度 (s[8]: rootx velocity)
  - 能量消耗 (torque * velocity)
  - 身体姿态 (angles of joints)
"""

# ========== Step 2: 状态空间语义标注 ==========
"""
Excel 配置 (mujoco_observation_space.xlsx → halfcheetah sheet):

Row | Index  | Description                     | Unit
----|--------|---------------------------------|----------
0   | s[0]   | z-coordinate of the front tip   | position (m)
1   | s[1]   | angle of the front tip          | angle (rad)
2-7 | s[2-7] | angles of 6 joints              | angle (rad)
8   | s[8]   | x-coordinate velocity           | velocity (m/s) ← 关键！
9   | s[9]   | z-coordinate velocity           | velocity (m/s)
10-16| s[10-16]| angular velocities of joints   | angular vel (rad/s)

解析后的 detail_content:
- `s[0]`: z-coordinate of the front tip , the unit is position (m).
- `s[1]`: angle of the front tip , the unit is angle (rad).
...
- `s[8]`: x-coordinate of the front tip , the unit is velocity (m/s).  ← 前向速度！
"""

# ========== Step 3: 任务描述撰写 ==========
task_description = """
The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them
(including two paws). The goal is to apply a torque on the joints to make the cheetah run forward
(right) as fast as possible, with a positive reward allocated based on the distance moved forward
and a negative reward allocated for moving backward. The torso and head of the cheetah are fixed,
and the torque can only be applied on the other 6 joints over the front and back thighs (connecting
to the torso), shins (connecting to the thighs) and feet (connecting to the shins).
"""

# 关键要素:
# ✅ 主体: 2D robot cheetah
# ✅ 自由度: 8 joints, 6 actuated
# ✅ 目标: run forward (right) as fast as possible
# ✅ 奖励: positive for forward, negative for backward

# ========== Step 4: 领域知识注入 ==========
"""
HalfCheetah 不需要 additional_prompt (为空)

原因:
- 不需要复杂协调 (相比 AntMaze 的四足协调)
- 目标直接 (前向速度)
- 不需要分阶段学习 (不像 Fetch 需要先学抓取再学放置)
"""

# ========== Step 5: 约束条件设计 ==========
"""
约束 1: 维度扩展
  - 输入: 27 维 (s[0] ~ s[26])
  - 输出: 必须 > 27 维
  - 推荐: 3-10 个额外维度

约束 2: intrinsic_reward 使用
  - 必须使用至少一个额外维度
  - 输出范围: [-100, 100]
  - 推荐: 组合多个特征 (速度、能量、稳定性)

约束 3: 数值稳定性
  - 避免除零: + epsilon
  - 避免 NaN/Inf: clip 到合理范围
"""

# ========== Step 6: 示例代码模板 ==========
"""
```python
import numpy as np
def revise_state(s):
    # Your state revision implementation goes here
    return updated_s
def intrinsic_reward(updated_s):
    # Your intrinsic reward code implementation goes here
    return intrinsic_reward
```
"""

# ========== Step 7: 迭代反馈设计 ==========
"""
性能指标: Final Policy Performance (累积奖励)

分析维度:
1. Lipschitz 常数: 哪些维度与奖励最相关
2. 性能对比: 最好样本 vs 最差样本
3. 特征有效性: 添加的维度是否发挥作用

改进建议:
- "Focus on forward velocity (s[8])"
- "Consider energy efficiency"
- "Reward smooth movements"
"""
```

### 5.3 实例：AntMaze 任务定制（对比）

```python
# ========== Step 1: 任务抽象理解 ==========
"""
任务类型: 目标导向 (到达指定位置)
关键物理量:
  - 位置 (x, y 坐标)
  - 四足协调 (4 条腿的关节)
  - 身体稳定性 (避免摔倒)
"""

# ========== Step 2: 状态空间语义标注 ==========
"""
AntMaze 状态空间 (27 维):
- s[0:2]: 身体位置 (x, y) ← 关键！
- s[2]: 身体高度 (z) ← 稳定性指标
- s[3:27]: 四个腿的关节角度和角速度
  - 每条腿: 3 joints × (angle + velocity) = 6 维
  - 4 条腿 × 6 = 24 维
"""

# ========== Step 3: 任务描述撰写 ==========
task_description = """
The agent is an ant-like robot with 4 legs, each leg has 3 joints. The goal is to navigate
through a maze to reach a target location. The ant needs to coordinate its four legs to
maintain balance while moving toward the goal.
"""

# ========== Step 4: 领域知识注入 (关键差异) ==========
"""
AntMaze 需要 additional_prompt:

additional_prompt = '''
Most Importantly: As for this task, the agent should firstly learn how to coordinate various
parts of the body itself before it finally reach the goal. Like a baby should learn how to walk
before finally walking to the goal. Therefore, when you design state representation and reward,
you should cosider how to make the agent learn to coordinate various parts of the body as well
as how to finally reach the goal.
'''

为什么需要这个提示？
1. 四足协调是先决条件
2. 直接奖励目标位置会导致摔倒
3. 需要分阶段学习 (先学走路，再学走向目标)
"""

# ========== Step 5-7: 约束、模板、反馈 ==========
"""
性能指标: Final Policy Success Rate (成功率，不是累积奖励)

分析维度:
1. 四足协调性 (腿部运动的同步性)
2. 身体稳定性 (z 轴高度变化)
3. 方向对齐 (速度方向与目标方向)

改进建议:
- "Reward diagonal coordination (对角线步态)"
- "Maintain body height (保持身体高度)"
- "Consider gait patterns (步态模式)"
"""
```

### 5.4 为什么要这样撰写？

| 撰写要素 | 原因 | 效果 |
|---------|------|------|
| **详尽的状态语义** | LLM 需要知道每个 s[i] 的物理意义 | 生成有物理意义的特征 (如前向速度、能量) |
| **单位信息** | 不同单位需要不同的缩放和约束 | 避免 m 和 rad 混用导致数值不稳定 |
| **任务描述具体化** | 抽象任务难以设计特征 | 具体目标引导特征设计 (如奖励前向速度) |
| **领域知识注入** | 利用 LLM 的世界知识 | AntMaze: 知道四足动物需要对角线步态 |
| **分阶段学习提示** | 复杂任务需要子目标 | Fetch: 先学抓取，再学放置 |
| **约束明确** | 避免 LLM 生成无效代码 | 维度检查、范围检查过滤无效样本 |
| **示例模板** | 降低格式错误率 | 减少"代码无法解析"的失败 |

## 六、提示词模板有效性检验方法

### 6.1 检验框架

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    提示词模板有效性检验四层金字塔                              │
└─────────────────────────────────────────────────────────────────────────────┘

                    Level 4: 任务性能提升
                    ┌───────────────────┐
                    │ 最终性能 vs Baseline│
                    │ 收敛速度           │
                    │ 样本效率           │
                    └─────────┬─────────┘
                              │
                    Level 3: 代码质量
                    ┌───────────────────┐
                    │ 验证通过率         │
                    │ 特征有效性         │
                    │ 数值稳定性         │
                    └─────────┬─────────┘
                              │
                    Level 2: LLM 输出质量
                    ┌───────────────────┐
                    │ 格式正确性         │
                    │ 约束满足率         │
                    │ 多样性             │
                    └─────────┬─────────┘
                              │
                    Level 1: 语义完整性
                    ┌───────────────────┐
                    │ 状态空间覆盖率     │
                    │ 任务描述清晰度     │
                    │ 约束明确性         │
                    └───────────────────┘
```

### 6.2 Level 1: 语义完整性检验

```python
def check_semantic_completeness(env_name, observation_path):
    """
    检验提示词的语义完整性
    """
    # 1. 状态空间覆盖率检查
    obs_file = pd.read_excel(observation_path, header=None, sheet_name=env_name)
    content = list(obs_file.iloc[:, 1])

    coverage_rate = len([c for c in content if not pd.isna(c)]) / len(content)
    print(f"状态空间覆盖率: {coverage_rate * 100:.1f}%")
    assert coverage_rate >= 0.95, "状态空间描述不完整"

    # 2. 任务描述清晰度检查
    task_description = obs_file.iloc[0, -1]
    assert len(task_description) > 50, "任务描述过于简短"
    assert 'goal' in task_description.lower(), "缺少目标描述"

    # 3. 单位信息完整性检查
    units = list(obs_file.iloc[:, -2])
    unit_coverage = len([u for u in units if not pd.isna(u)]) / len(units)
    print(f"单位信息覆盖率: {unit_coverage * 100:.1f}%")

    # 4. 约束明确性检查
    required_constraints = [
        "维度范围",
        "输出范围 [-100, 100]",
        "必须使用额外维度"
    ]
    for constraint in required_constraints:
        assert constraint in prompt, f"缺少约束: {constraint}"

    return {
        'state_coverage': coverage_rate,
        'task_description_length': len(task_description),
        'unit_coverage': unit_coverage
    }

# ========== 示例输出 ==========
"""
检验结果:
✅ 状态空间覆盖率: 100.0%
✅ 任务描述长度: 587 字符
✅ 单位信息覆盖率: 100.0%
✅ 约束明确性: 所有必需约束均存在
"""
```

### 6.3 Level 2: LLM 输出质量检验

```python
def check_llm_output_quality(sample_count=10):
    """
    检验 LLM 输出的质量
    """
    format_correct_count = 0
    constraint_satisfied_count = 0
    diversity_scores = []

    for i in range(sample_count):
        # 生成样本
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        code = response['choices'][0]['message']['content']

        # 1. 格式正确性
        try:
            extracted_code = extract_python_code(code)
            compile(extracted_code, '<string>', 'exec')
            format_correct_count += 1
        except:
            pass

        # 2. 约束满足率
        try:
            module = load_module(extracted_code)
            test_state = np.random.rand(state_dim)
            revised_state = module.revise_state(test_state)

            # 检查维度
            if len(revised_state) > state_dim:
                constraint_satisfied_count += 1

            # 检查 intrinsic_reward
            reward = module.intrinsic_reward(revised_state)
            assert -100 <= reward <= 100
        except:
            pass

        # 3. 多样性 (代码编辑距离)
        diversity_scores.append(calculate_code_diversity(extracted_code))

    # 统计结果
    format_correct_rate = format_correct_count / sample_count
    constraint_satisfied_rate = constraint_satisfied_count / sample_count
    avg_diversity = np.mean(diversity_scores)

    print(f"格式正确率: {format_correct_rate * 100:.1f}%")
    print(f"约束满足率: {constraint_satisfied_rate * 100:.1f}%")
    print(f"平均多样性分数: {avg_diversity:.2f}")

    return {
        'format_correct_rate': format_correct_rate,
        'constraint_satisfied_rate': constraint_satisfied_rate,
        'diversity': avg_diversity
    }

# ========== 示例输出 ==========
"""
LLM 输出质量检验:
✅ 格式正确率: 90.0% (目标: >85%)
✅ 约束满足率: 75.0% (目标: >70%)
✅ 平均多样性分数: 0.68 (目标: >0.5)
"""
```

### 6.4 Level 3: 代码质量检验

```python
def check_code_quality(code_samples, test_state):
    """
    检验生成代码的质量
    """
    validation_pass_rate = 0
    feature_effectiveness_scores = []
    numerical_stability_scores = []

    for code in code_samples:
        try:
            # 1. 验证通过率
            module = load_module(code)
            revised_state = module.revise_state(test_state)
            reward = module.intrinsic_reward(revised_state)

            # 检查数值稳定性
            if not np.any(np.isnan(revised_state)) and not np.any(np.isinf(revised_state)):
                validation_pass_rate += 1

            # 2. 特征有效性 (Lipschitz 分析)
            lipschitz_constants = compute_lipschitz(
                module.revise_state,
                test_state_samples
            )
            # 添加维度的 Lipschitz 常数应该 > 0
            added_dims_lipschitz = lipschitz_constants[state_dim:]
            feature_effectiveness = np.mean(added_dims_lipschitz > 0.01)
            feature_effectiveness_scores.append(feature_effectiveness)

            # 3. 数值稳定性 (极值测试)
            extreme_states = np.random.uniform(-100, 100, (100, state_dim))
            stability_score = 0
            for s in extreme_states:
                try:
                    r = module.intrinsic_reward(module.revise_state(s))
                    if -100 <= r <= 100:
                        stability_score += 1
                except:
                    pass
            numerical_stability_scores.append(stability_score / 100)

        except Exception as e:
            pass

    # 统计结果
    validation_pass_rate /= len(code_samples)
    avg_feature_effectiveness = np.mean(feature_effectiveness_scores)
    avg_numerical_stability = np.mean(numerical_stability_scores)

    print(f"验证通过率: {validation_pass_rate * 100:.1f}%")
    print(f"平均特征有效性: {avg_feature_effectiveness:.2f}")
    print(f"平均数值稳定性: {avg_numerical_stability * 100:.1f}%")

    return {
        'validation_pass_rate': validation_pass_rate,
        'feature_effectiveness': avg_feature_effectiveness,
        'numerical_stability': avg_numerical_stability
    }

# ========== 示例输出 ==========
"""
代码质量检验:
✅ 验证通过率: 85.0% (目标: >80%)
✅ 平均特征有效性: 0.72 (目标: >0.5)
✅ 平均数值稳定性: 95.0% (目标: >90%)
"""
```

### 6.5 Level 4: 任务性能提升检验

```python
def check_task_performance(baseline_score, best_lesr_score, iteration_scores):
    """
    检验任务性能提升
    """
    # 1. 最终性能提升
    improvement_rate = (best_lesr_score - baseline_score) / baseline_score
    print(f"性能提升率: {improvement_rate * 100:.1f}%")

    # 2. 收敛速度 (达到 90% 最终性能所需的迭代数)
    target_score = baseline_score + 0.9 * (best_lesr_score - baseline_score)
    convergence_iter = 0
    for i, score in enumerate(iteration_scores):
        if score >= target_score:
            convergence_iter = i + 1
            break
    print(f"收敛迭代数: {convergence_iter}")

    # 3. 样本效率 (每轮平均提升)
    avg_improvement_per_iter = np.mean(np.diff(iteration_scores))
    print(f"平均每轮提升: {avg_improvement_per_iter:.2f}")

    # 4. 稳定性 (性能方差)
    score_variance = np.var(iteration_scores)
    print(f"性能方差: {score_variance:.2f}")

    return {
        'improvement_rate': improvement_rate,
        'convergence_iter': convergence_iter,
        'avg_improvement_per_iter': avg_improvement_per_iter,
        'score_variance': score_variance
    }

# ========== 示例输出 ==========
"""
任务性能提升检验:
✅ 性能提升率: 35.2% (目标: >20%)
✅ 收敛迭代数: 3 (目标: ≤5)
✅ 平均每轮提升: 525.3 (目标: >300)
✅ 性能方差: 12450.2 (目标: <20000)
"""
```

### 6.6 综合有效性评分

```python
def calculate_overall_effectiveness_score(
    semantic_check,
    llm_output_check,
    code_quality_check,
    task_performance_check
):
    """
    计算综合有效性评分 (0-100)
    """
    # 权重分配
    weights = {
        'semantic': 0.15,        # 语义完整性是基础
        'llm_output': 0.25,      # LLM 输出质量决定效率
        'code_quality': 0.30,    # 代码质量决定可行性
        'task_performance': 0.30 # 任务性能是最终目标
    }

    # Level 1: 语义完整性 (15%)
    semantic_score = (
        semantic_check['state_coverage'] * 0.4 +
        (semantic_check['task_description_length'] / 1000) * 0.3 +
        semantic_check['unit_coverage'] * 0.3
    ) * 100

    # Level 2: LLM 输出质量 (25%)
    llm_output_score = (
        llm_output_check['format_correct_rate'] * 0.3 +
        llm_output_check['constraint_satisfied_rate'] * 0.4 +
        min(llm_output_check['diversity'], 1.0) * 0.3
    ) * 100

    # Level 3: 代码质量 (30%)
    code_quality_score = (
        code_quality_check['validation_pass_rate'] * 0.3 +
        code_quality_check['feature_effectiveness'] * 0.4 +
        code_quality_check['numerical_stability'] * 0.3
    ) * 100

    # Level 4: 任务性能 (30%)
    task_performance_score = (
        min(task_performance_check['improvement_rate'] * 2, 1.0) * 0.4 +
        (1 - task_performance_check['convergence_iter'] / 10) * 0.2 +
        min(task_performance_check['avg_improvement_per_iter'] / 1000, 1.0) * 0.2 +
        max(1 - task_performance_check['score_variance'] / 50000, 0) * 0.2
    ) * 100

    # 综合评分
    overall_score = (
        semantic_score * weights['semantic'] +
        llm_output_score * weights['llm_output'] +
        code_quality_score * weights['code_quality'] +
        task_performance_score * weights['task_performance']
    )

    print("=" * 60)
    print("提示词模板有效性综合评分")
    print("=" * 60)
    print(f"Level 1 - 语义完整性:     {semantic_score:.1f}/100 (权重 15%)")
    print(f"Level 2 - LLM 输出质量:   {llm_output_score:.1f}/100 (权重 25%)")
    print(f"Level 3 - 代码质量:       {code_quality_score:.1f}/100 (权重 30%)")
    print(f"Level 4 - 任务性能提升:   {task_performance_score:.1f}/100 (权重 30%)")
    print("=" * 60)
    print(f"综合评分: {overall_score:.1f}/100")
    print("=" * 60)

    # 评级
    if overall_score >= 90:
        grade = "A+ (优秀)"
    elif overall_score >= 80:
        grade = "A (良好)"
    elif overall_score >= 70:
        grade = "B (中等)"
    elif overall_score >= 60:
        grade = "C (及格)"
    else:
        grade = "D (需要改进)"

    print(f"评级: {grade}")

    return {
        'overall_score': overall_score,
        'grade': grade,
        'breakdown': {
            'semantic': semantic_score,
            'llm_output': llm_output_score,
            'code_quality': code_quality_score,
            'task_performance': task_performance_score
        }
    }

# ========== 示例输出 ==========
"""
============================================================
提示词模板有效性综合评分
============================================================
Level 1 - 语义完整性:     95.0/100 (权重 15%)
Level 2 - LLM 输出质量:   82.5/100 (权重 25%)
Level 3 - 代码质量:       88.3/100 (权重 30%)
Level 4 - 任务性能提升:   91.2/100 (权重 30%)
============================================================
综合评分: 88.6/100
============================================================
评级: A (良好)
"""
```

### 6.7 快速检验清单（适用于新任务）

```python
def quick_validation_checklist(env_name, prompt_template):
    """
    快速检验清单（5 分钟快速验证）
    """
    print("=== 提示词模板快速检验清单 ===")

    checks = {
        "✅ 任务描述包含目标": "goal" in prompt_template.lower(),
        "✅ 状态空间描述完整": "s[0]" in prompt_template and "s[" in prompt_template,
        "✅ 单位信息存在": "unit is" in prompt_template.lower(),
        "✅ 维度约束明确": "dimensional" in prompt_template.lower(),
        "✅ 输出范围限制": "[-100, 100]" in prompt_template,
        "✅ 必须使用额外维度": "must use the extra dim" in prompt_template.lower(),
        "✅ 函数签名固定": "def revise_state(s)" in prompt_template,
        "✅ 示例代码存在": "```python" in prompt_template,
        "✅ 分析引导提示": "(a)" in prompt_template or "(b)" in prompt_template,
        "✅ Lipschitz 分析": "lipschitz" in prompt_template.lower()
    }

    pass_count = sum(checks.values())
    total_count = len(checks)

    for check, result in checks.items():
        print(f"{check}: {'✓' if result else '✗'}")

    print(f"\n通过率: {pass_count}/{total_count} ({pass_count/total_count*100:.1f}%)")

    if pass_count == total_count:
        print("🎉 所有检验通过！提示词模板准备就绪。")
    elif pass_count >= total_count * 0.8:
        print("⚠️  大部分检验通过，建议优化缺失项。")
    else:
        print("❌ 检验未通过，提示词模板需要大幅改进。")

    return checks

# ========== 示例输出 ==========
"""
=== 提示词模板快速检验清单 ===
✅ 任务描述包含目标: ✓
✅ 状态空间描述完整: ✓
✅ 单位信息存在: ✓
✅ 维度约束明确: ✓
✅ 输出范围限制: ✓
✅ 必须使用额外维度: ✓
✅ 函数签名固定: ✓
✅ 示例代码存在: ✓
✅ 分析引导提示: ✓
✅ Lipschitz 分析: ✓

通过率: 10/10 (100.0%)
🎉 所有检验通过！提示词模板准备就绪。
"""
```

## 七、总结：通用框架迁移指南

### 7.1 通用框架可迁移组件

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    可直接迁移到新任务的组件                                     │
└─────────────────────────────────────────────────────────────────────────────┘

1. Prompt 模板结构
   ├── 初始化 Prompt 骨架 (init_prompt_template)
   ├── COT 反馈 Prompt 骨架 (cot_prompt_template)
   └── 迭代改进 Prompt 骨架 (next_iteration_prompt_template)

2. 代码验证框架
   ├── extract_python_code() - 代码提取
   ├── validate_functions() - 功能验证
   ├── check_dimensions() - 维度检查
   └── check_numerical_stability() - 数值稳定性检查

3. 反馈分析框架
   ├── compute_lipschitz() - Lipschitz 常数计算
   ├── rank_by_performance() - 性能排序
   ├── identify_best_worst() - 最好/最差识别
   └── generate_feedback_text() - 反馈文本生成

4. 迭代优化循环
   ├── sample_state_revision_functions() - 采样
   ├── parallel_training() - 并行训练
   ├── analyze_and_feedback() - 分析反馈
   └── update_prompt() - 更新 Prompt

5. 输出格式规范
   ├── Python 函数签名
   ├── 代码块格式 (```python)
   └── 返回值类型约束
```

### 7.2 任务定制必需组件

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    需要针对新任务定制的组件                                     │
└─────────────────────────────────────────────────────────────────────────────┘

1. 状态空间语义描述 (Excel 配置)
   ├── 每个维度的物理含义
   ├── 单位信息 (m, rad, m/s, rad/s)
   ├── 取值范围 (如有)
   └── 关键维度标注

2. 任务描述 (Task Description)
   ├── 主体是什么 (机器人/智能体/系统)
   ├── 环境特征 (2D/3D, 地形, 约束)
   ├── 目标是什么 (前向运动/到达目标/优化指标)
   └── 奖励机制 (距离/速度/成功率/效率)

3. 领域特定提示 (Additional Prompt)
   ├── 先验知识 (身体协调/步态模式/物理定律)
   ├── 分阶段学习 (子目标/课程学习)
   ├── 常见失败模式 (摔倒/震荡/局部最优)
   └── 关键成功因素 (平衡/效率/方向)

4. 性能指标语义
   ├── Success Rate (迷宫/抓取任务)
   ├── Cumulative Reward (运动/控制任务)
   ├── Efficiency (能量/时间)
   └── Stability (方差/平滑度)
```

### 7.3 迁移流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    将 LESR 框架迁移到新任务的步骤                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: 分析新任务
├─ 任务类型? (运动/导航/操纵/调度)
├─ 状态空间维度?
├─ 关键物理量?
└─ 目标是什么?

Step 2: 创建状态空间配置 (Excel)
├─ 列出所有状态维度
├─ 描述物理含义
├─ 标注单位
└─ 标记关键维度

Step 3: 撰写任务描述
├─ 主体: 什么系统?
├─ 目标: 优化什么?
├─ 约束: 有哪些限制?
└─ 奖励: 如何评估?

Step 4: 设计领域提示 (可选)
├─ 需要先验知识吗?
├─ 需要分阶段学习吗?
├─ 有常见失败模式吗?
└─ 有关键成功因素吗?

Step 5: 配置通用框架
├─ 复制 Prompt 模板
├─ 填入任务定制内容
├─ 调整性能指标
└─ 设置约束条件

Step 6: 快速验证
├─ 运行 quick_validation_checklist()
├─ 生成 10 个样本
├─ 检查格式正确率
└─ 检查约束满足率

Step 7: 小规模实验
├─ 运行 1-2 轮迭代
├─ 验证代码可训练
├─ 检查性能趋势
└─ 调整 Prompt

Step 8: 完整运行
├─ 5-10 轮迭代
├─ 对比 Baseline
└─ 评估提升率
```

---

**文档版本**: v1.0
**最后更新**: 2026-04-02
**作者**: LESR 提示词机制深度分析
**相关文档**: [LLM代码生成机制详解.md](../../参考项目梳理/LESR/LLM代码生成机制详解.md)
