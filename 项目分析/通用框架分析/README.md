# LESR 语义提示词模块 - 文档索引

## 📚 文档概览

本模块提供了 LESR (LLM-Empowered State Representation for RL) 项目中 LLM 提示词机制的全面分析，包括通用框架、任务定制策略、实战案例和实用工具。

### 📖 文档结构

```
语义提示词模块/
├── README.md (本文件)
│   └── 快速入门和文档索引
│
├── 01-LESR提示词机制架构分析.md
│   ├── 通用框架 vs 任务定制
│   ├── 提示词设计三角原则
│   ├── 三种 Prompt 类型详解
│   └── 有效性检验四层金字塔
│
├── 02-任务定制提示词撰写实战.md
│   ├── 任务分类与定制策略矩阵
│   ├── HalfCheetah 实战案例 (运动任务)
│   ├── AntMaze 实战案例 (导航任务)
│   ├── 两种任务对比分析
│   └── 提示词优化迭代流程
│
└── 03-提示词模板工具集.md
    ├── PromptTemplateGenerator (提示词生成器)
    ├── CodeValidator (代码验证工具)
    ├── PromptEvaluator (有效性评估工具)
    └── LESRPromptToolkit (完整工具集)
```

## 🚀 快速入门

### 5 分钟快速理解 LESR 提示词机制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LESR 提示词机制核心概念                                     │
└─────────────────────────────────────────────────────────────────────────────┘

核心思想:
├── LLM 作为"状态表示设计师"
├── 自动生成 revise_state() 和 intrinsic_reward() 函数
├── 通过训练反馈迭代优化
└── 最终找到最优状态表示

三大 Prompt 类型:
├── 初始化 Prompt: 首次生成代码
├── COT 反馈 Prompt: 分析训练结果
└── 迭代改进 Prompt: 基于历史经验优化

通用框架 (可迁移):
├── Prompt 模板结构
├── 代码验证逻辑
├── 反馈分析框架
└── 迭代优化循环

任务定制 (需配置):
├── 状态空间语义描述 (Excel)
├── 任务描述 (Task Description)
├── 领域特定提示 (Additional Prompt)
└── 性能指标语义 (Success Rate vs Reward)
```

### 10 分钟快速上手

```python
# ========== Step 1: 安装依赖 ==========
# pip install pandas numpy openai

# ========== Step 2: 创建工具集 ==========
from lesr_prompt_toolkit import LESRPromptToolkit

toolkit = LESRPromptToolkit(
    observation_path="LESR-resources/mujoco_observation_space.xlsx"
)

# ========== Step 3: 为任务创建提示词 ==========
config, init_prompt = toolkit.create_prompt_for_task("HalfCheetah-v4")

# ========== Step 4: 快速验证提示词 ==========
validation_results = toolkit.quick_validate(init_prompt)

# ========== Step 5: 调用 LLM 生成代码 ==========
import openai

response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[{"role": "user", "content": init_prompt}],
    temperature=0.7
)
generated_code = response['choices'][0]['message']['content']

# ========== Step 6: 验证生成的代码 ==========
codes = [generated_code]  # 可以生成多个
validation_results = toolkit.validate_generated_codes(codes, "HalfCheetah-v4")

# ========== Step 7: 训练和评估 ==========
# (使用 LESR 训练框架训练)

# ========== Step 8: 评估提示词有效性 ==========
overall = toolkit.evaluate_prompt_effectiveness(
    env_name="HalfCheetah-v4",
    prompt=init_prompt,
    generated_codes=codes,
    validation_results=validation_results,
    iteration_scores=[3500, 4200, 4800, 5100, 5350],
    baseline_score=3300
)

print(f"综合评分: {overall['overall_score']:.1f}/100")
print(f"评级: {overall['grade']}")
```

## 📊 核心概念速查表

### 任务类型分类

| 任务类型 | 特征 | 关键物理量 | 需要领域提示 | 性能指标 |
|---------|------|-----------|-------------|---------|
| **运动任务** (HalfCheetah) | 最大化速度 | 前向速度、能量、稳定性 | 否 | Cumulative Reward |
| **导航任务** (AntMaze) | 到达目标 | 距离、方向、身体协调 | 是 (必需) | Success Rate |
| **操作任务** (Fetch) | 操作物体 | 抓取力、物体位置 | 是 (必需) | Success Rate |

### 提示词组件速查

| 组件 | 通用/定制 | 说明 |
|-----|----------|------|
| 任务描述 | ✅ 定制 | 描述任务目标、环境、奖励机制 |
| 状态空间语义 | ✅ 定制 | 每个维度的物理含义和单位 |
| 维度约束 | ✅ 通用 | 输入输出规格、范围限制 |
| 领域提示 | ✅ 定制 | 先验知识、分阶段学习 |
| 函数签名 | ✅ 通用 | `def revise_state(s)` 和 `def intrinsic_reward(updated_s)` |
| 示例代码 | ✅ 通用 | Python 代码模板 |
| COT 分析 | ✅ 通用 | Lipschitz 分析、性能对比 |
| 反馈引导 | ✅ 通用 | (a)(b)(c) 分析提示 |

### 有效性检验四层

```
Level 1: 语义完整性 (15%)
├── 状态空间覆盖率
├── 任务描述长度
└── 单位信息完整性

Level 2: LLM 输出质量 (25%)
├── 格式正确率
├── 约束满足率
└── 多样性

Level 3: 代码质量 (30%)
├── 验证通过率
├── 特征有效性
└── 数值稳定性

Level 4: 任务性能提升 (30%)
├── 性能提升率
├── 收敛速度
└── 样本效率
```

## 🎯 常见使用场景

### 场景 1: 为新任务创建提示词

```python
# 1. 准备 Excel 配置文件
#    - 描述每个状态维度
#    - 标注单位和物理含义
#    - 提供任务描述

# 2. 使用工具集创建提示词
toolkit = LESRPromptToolkit(observation_path="your_config.xlsx")
config, prompt = toolkit.create_prompt_for_task("YourTask-v0")

# 3. 快速验证
validation = toolkit.quick_validate(prompt)

# 4. 手动检查和调整
#    - 检查任务描述是否准确
#    - 检查关键维度是否标注
#    - 添加必要的领域提示
```

### 场景 2: 优化现有提示词

```python
# 1. 运行一轮迭代
#    - 生成代码样本
#    - 训练并收集结果

# 2. 分析 COT 反馈
cot_prompt = toolkit.generator.generate_cot_prompt(
    config=config,
    training_results=results
)

# 3. 识别问题模式
#    - 哪些特征有效？
#    - 哪些特征无效？
#    - 常见失败模式？

# 4. 调整提示词
#    - 强调成功模式
#    - 避免失败模式
#    - 添加具体建议

# 5. 生成下一轮提示词
next_prompt = toolkit.generator.generate_iteration_prompt(
    config=config,
    history=iteration_history
)
```

### 场景 3: 评估提示词质量

```python
# 1. 收集数据
#    - 提示词
#    - 生成的代码
#    - 验证结果
#    - 训练性能
#    - 基线性能

# 2. 全面评估
overall = toolkit.evaluate_prompt_effectiveness(
    env_name="YourTask",
    prompt=prompt,
    generated_codes=codes,
    validation_results=validation_results,
    iteration_scores=scores,
    baseline_score=baseline
)

# 3. 分析结果
print(f"综合评分: {overall['overall_score']:.1f}/100")
print(f"评级: {overall['grade']}")
print(f"语义完整性: {overall['breakdown']['semantic']:.1f}/100")
print(f"LLM 输出质量: {overall['breakdown']['llm_output']:.1f}/100")
print(f"代码质量: {overall['breakdown']['code_quality']:.1f}/100")
print(f"任务性能: {overall['breakdown']['task_performance']:.1f}/100")

# 4. 针对性改进
#    - 识别薄弱环节
#    - 调整提示词
#    - 重新评估
```

## 🔧 实用工具清单

### PromptTemplateGenerator

```python
# 功能：自动生成提示词模板

# 主要方法:
├── load_task_config(env_name)          # 加载任务配置
├── generate_init_prompt(config)        # 生成初始化 Prompt
├── generate_cot_prompt(config, results) # 生成 COT 反馈 Prompt
└── generate_iteration_prompt(config, history) # 生成迭代 Prompt

# 自动检测:
├── 任务类型 (运动/导航/操作)
├── 性能指标 (奖励/成功率)
├── 关键维度
└── 是否需要领域提示
```

### CodeValidator

```python
# 功能：验证生成的代码

# 主要方法:
├── extract_python_code(llm_output)     # 提取 Python 代码
├── validate_code(code, test_state, original_dim) # 验证单个代码
└── batch_validate(codes, test_state, original_dim) # 批量验证

# 检查项:
├── 代码可执行性
├── 维度约束 (输出 > 输入)
├── 数值稳定性 (无 NaN/Inf)
└── 输出范围 ([-100, 100])
```

### PromptEvaluator

```python
# 功能：评估提示词有效性

# 主要方法:
├── check_semantic_completeness(config, prompt) # 检查语义完整性
├── check_llm_output_quality(codes, results)    # 检查 LLM 输出质量
├── check_code_quality(validation_results)      # 检查代码质量
├── check_task_performance(scores, baseline)    # 检查任务性能
└── calculate_overall_score(...)                # 计算综合评分

# 评估层次:
├── Level 1: 语义完整性 (15%)
├── Level 2: LLM 输出质量 (25%)
├── Level 3: 代码质量 (30%)
└── Level 4: 任务性能 (30%)
```

### LESRPromptToolkit

```python
# 功能：端到端工具集

# 主要方法:
├── create_prompt_for_task(env_name)              # 创建提示词
├── validate_generated_codes(codes, env_name)     # 验证代码
├── evaluate_prompt_effectiveness(...)            # 评估有效性
└── quick_validate(prompt)                        # 快速验证

# 使用流程:
├── Step 1: 创建工具集
├── Step 2: 为任务创建提示词
├── Step 3: 快速验证提示词
├── Step 4: 生成代码样本
├── Step 5: 验证生成的代码
├── Step 6: 训练和评估
├── Step 7: 评估提示词有效性
└── Step 8: 迭代优化
```

## 📖 深入阅读指南

### 如果你想理解架构...

👉 阅读 **[01-LESR提示词机制架构分析.md](./01-LESR提示词机制架构分析.md)**

重点章节:
- 第二章：通用框架 vs 任务定制
- 第三章：提示词设计的核心理念
- 第四章：三种 Prompt 的详细分析

### 如果你想实战定制...

👉 阅读 **[02-任务定制提示词撰写实战.md](./02-任务定制提示词撰写实战.md)**

重点章节:
- 第一章：任务分类与定制策略矩阵
- 第二章：HalfCheetah 实战案例
- 第三章：AntMaze 实战案例
- 第六章：提示词撰写清单

### 如果你想使用工具...

👉 阅读 **[03-提示词模板工具集.md](./03-提示词模板工具集.md)**

重点章节:
- 第一章：提示词模板生成器
- 第二章：代码验证工具
- 第三章：有效性评估工具
- 第二章：完整的端到端工具

## 💡 关键要点总结

### 通用框架的核心特征

```
✅ 任务无关的组件:
├── Prompt 模板结构 (占位符系统)
├── 代码验证逻辑 (维度、数值、范围检查)
├── 反馈分析框架 (Lipschitz、性能排序)
├── 迭代优化循环 (COT 反馈 → 改进)
└── 输出格式规范 (Python 函数签名)

✅ 可直接迁移到新任务:
├── 复制 Prompt 模板
├── 填入任务定制内容
├── 调整性能指标
└── 设置约束条件
```

### 任务定制的核心要素

```
✅ 需要针对具体任务配置:
├── 状态空间语义描述 (Excel 配置文件)
├── 任务描述 (Task Description)
├── 领域特定提示 (Additional Prompt)
├── 性能指标语义 (Success Rate vs Cumulative Reward)
└── 物理约束和单位 (m, rad, m/s, rad/s)

✅ 决定是否需要领域提示:
├── 运动任务 (HalfCheetah): 通常不需要
├── 导航任务 (AntMaze): 必需！
├── 操作任务 (Fetch): 必需！
└── 判断标准: 是否需要分阶段学习
```

### 有效性检验的黄金标准

```
✅ 综合评分 ≥ 80 分: 提示词有效
✅ 综合评分 ≥ 90 分: 提示词优秀
✅ 综合评分 < 70 分: 需要改进

✅ 四层平衡:
├── 语义完整性: >90%
├── LLM 输出质量: >75%
├── 代码质量: >80%
└── 任务性能: >20% 提升
```

## 🚨 常见问题和解决方案

### Q1: 如何判断我的任务是否需要领域提示？

```
判断标准:
1. 是否需要身体协调？ (AntMaze: 四足协调)
2. 是否需要分阶段学习？ (Fetch: 先抓取再放置)
3. 是否有常见失败模式？ (摔倒、震荡)

如果以上任何一项为"是"，则需要添加领域提示。
```

### Q2: 如何识别关键状态维度？

```
方法:
1. 查看任务描述中的关键词
   - HalfCheetah: "forward" → s[8] (x 方向速度)
   - AntMaze: "goal" → s[0:2] (位置)

2. 查看状态维度的单位
   - velocity (m/s) → 运动相关
   - position (m) → 位置相关
   - angle (rad) → 姿态相关

3. 查看 Lipschitz 分析结果
   - 高相关性 → 关键维度
```

### Q3: 提示词验证通过率低怎么办？

```
问题诊断:
1. 格式正确率低 → 检查示例代码是否清晰
2. 约束满足率低 → 检查约束条件是否明确
3. 数值不稳定 → 添加 epsilon 和 clip 提示

解决方案:
1. 在 Prompt 中强调约束条件
2. 提供更多示例代码
3. 添加常见错误警告
```

### Q4: 如何提高任务性能提升？

```
优化策略:
1. 分析 COT 反馈
   - 识别成功模式
   - 避免失败模式

2. 调整特征重点
   - 强调高 Lipschitz 维度
   - 移除无关特征

3. 优化奖励设计
   - 平衡多个目标
   - 分阶段学习 (如需要)

4. 迭代改进
   - 每轮基于历史经验
   - 渐进式优化
```

### Q5: 如何迁移到新任务？

```
迁移流程:
1. 创建 Excel 配置文件
   - 描述状态空间
   - 标注单位和含义

2. 使用工具集
   - toolkit = LESRPromptToolkit(...)
   - config, prompt = toolkit.create_prompt_for_task(...)

3. 快速验证
   - validation = toolkit.quick_validate(prompt)

4. 小规模实验
   - 运行 1-2 轮迭代
   - 检查性能趋势

5. 完整运行
   - 5-10 轮迭代
   - 评估提升率
```

## 📞 获取帮助

### 文档内搜索

使用关键词搜索:
- "通用框架" - 查看可迁移的组件
- "任务定制" - 查看需要配置的部分
- "HalfCheetah" - 查看运动任务案例
- "AntMaze" - 查看导航任务案例
- "有效性检验" - 查看评估方法
- "工具" - 查看代码实现

### 代码示例

所有文档都包含完整的代码示例:
- 复制粘贴即可运行
- 注释详细
- 涵盖常见场景

### 实战案例

两个完整的实战案例:
1. HalfCheetah (运动任务)
2. AntMaze (导航任务)

对比分析，理解差异。

## 📝 版本历史

- **v1.0** (2026-04-02)
  - 初始版本
  - 包含三个核心文档
  - 提供完整的工具集

## 🔗 相关资源

### 内部资源
- [LLM代码生成机制详解.md](../../参考项目梳理/LESR/LLM代码生成机制详解.md)
- [LESR系统架构分析.md](../../参考项目梳理/LESR/LESR系统架构分析.md)

### 外部资源
- [LESR GitHub Repository](https://github.com/LESR-project/LESR)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [MuJoCo Physics Engine](https://mujoco.org/)

---

**最后更新**: 2026-04-02
**维护者**: LESR 项目分析团队
**许可证**: MIT License
