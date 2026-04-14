# 系统架构

## 系统概览

LESR（LLM 赋能的状态表示）— 一个研究框架，利用 LLM 为基于强化学习的股票交易生成特征工程函数。系统通过迭代方式改进状态表示：从 LLM 采样 Python 代码 → 训练 DQN 智能体 → 分析特征重要性 → 将洞察反馈给 LLM。

## 核心管道（exp4.7）

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  YAML 配置   │───>│  LESR 循环   │───>│  DQN 训练器  │───>│  特征分析器  │
│  + 数据      │    │  控制器      │    │  (每个样本)  │    │              │
└─────────────┘    └──────┬───────┘    └──────────────┘    └──────┬───────┘
                          │                                        │
                   ┌──────▼───────┐                         ┌──────▼───────┐
                   │  LLM 采样器  │                         │  COT 反馈    │
                   │  (OpenAI)    │                         │  生成        │
                   └──────────────┘                         └──────────────┘
```

## 数据流

1. **配置加载** → `main.py` 读取 YAML 配置和 pickle 数据
2. **LESR 循环**（`lesr_controller.py`）：
   - 第 0 轮迭代：生成初始 prompt → LLM 产出 `revise_state()` + `intrinsic_reward()` 代码
   - 每轮迭代：
     a. 解析 LLM 输出为可执行 Python 模块（临时文件）
     b. 为每只股票启动并行 DQN 训练 worker
     c. 收集训练指标（回合奖励、状态）
     d. 运行特征重要性分析（相关性 + SHAP）
     e. 生成 COT（思维链）反馈 prompt
     f. LLM 生成下一轮迭代的代码
3. **最优选择** → 按验证集夏普比率挑选最佳样本
4. **回测** → 在测试期运行 FINSABER 回测
5. **基线对比** → 训练原始 DQN（无特征工程）作为对照

## 关键抽象

### LESRController（`exp4.7/lesr_controller.py`）
- 编排整个优化循环
- 管理迭代、LLM 调用、并行 worker 启动
- 使用 `torch.multiprocessing.Pool` 进行多 GPU 训练
- 通过 `importlib` 动态导入 LLM 生成的代码

### DQNTrainer（`exp4.7/dqn_trainer.py`）
- 标准 DQN：经验回放缓冲区、目标网络、epsilon-贪心策略
- 接受 `revise_state_func` 和 `intrinsic_reward_func` 作为可调用对象
- 输出：训练好的 DQN 模型、回合状态/奖励用于分析

### 特征分析器（`exp4.7/feature_analyzer.py`）
- 替代原始 LESR 的 Lipschitz 分析（不适用于金融数据）
- 使用 Spearman 相关性 + SHAP 值
- 识别哪些 LLM 生成的特征真正有贡献

### Prompt 模板（`exp4.7/prompts.py`）
- `INITIAL_PROMPT` — 描述状态结构、金融概念、特征库
- `get_iteration_prompt()` — 包含前一轮的 COT 反馈
- `get_financial_cot_prompt()` — 为 LLM 生成分析反馈

## 入口脚本

| 脚本 | 用途 |
|------|------|
| `exp4.7/main.py` | 完整管道：配置 → LESR 循环 → 回测 → 对比 |
| `exp4.7/main_simple.py` | 简化单次运行版本 |
| `exp4.7/run_amzn_nflx.py` | AMZN/NFLX 专用运行 |
| `exp4.7/run_window.py` | 滑动窗口实验（W1-W10） |
| `exp4.7/run_4stocks_2012_2017.py` | 多股票长周期运行 |

## 重构架构（llm_rl_trading_finsaber/）

更模块化的重写版本，具有规范的包结构：
- `src/lesr/` — LLM 采样器、prompt 模板、修订候选
- `src/drl/` — DQN/TD3 运行器、策略、回放缓冲区、指标
- `src/env/` — 交易环境（Gym、FINSABER 兼容）
- `src/data/` — 数据加载、特征工程
- `src/llm/` — 多 LLM 客户端支持（OpenAI、DeepSeek）
- `src/pipeline/` — 分支迭代、市场状态专家、日期过滤
- `src/finsaber_native/` — 原生 FINSABER 集成（环境、模型、预处理器）
