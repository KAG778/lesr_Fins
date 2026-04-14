# 目录结构

## 目录布局

```
lesr/                                 # 项目根目录
├── exp4.7/                           # ★ 当前主要活跃实验
│   ├── lesr_controller.py            # LESR 优化主循环
│   ├── dqn_trainer.py                # DQN 训练（ReplayBuffer + DQN + DQNTrainer）
│   ├── feature_analyzer.py           # 特征重要性（Spearman + SHAP）
│   ├── prompts.py                    # LLM prompt 模板
│   ├── baseline.py                   # 基线 DQN（原始特征，无 LLM）
│   ├── lesr_strategy.py              # LESR 策略包装器（用于 FINSABER）
│   ├── main.py                       # 完整管道入口
│   ├── main_simple.py                # 简化入口
│   ├── run_amzn_nflx.py              # AMZN/NFLX 专用运行脚本
│   ├── run_window.py                 # 滑动窗口实验
│   ├── run_4stocks_2012_2017.py      # 多股票长周期运行
│   ├── prepare_data.py               # 数据准备工具
│   ├── config.yaml                   # 默认配置
│   ├── config_W{1..10}.yaml          # 各时间窗口配置
│   ├── config_nflx.yaml              # NFLX 专用配置
│   ├── requirements.txt              # Python 依赖
│   └── logs/                         # 实验日志
│
├── 4.12exp/                          # 最新实验变体
├── exp4.9/                           # 早期实验变体
├── exp_4.9_b/                        # 变体 B
├── exp4.9_c/                         # 变体 C（滑动窗口）
├── exp4.9_d/                         # 变体 D
├── EXP4.9_f/                         # 变体 F
│
├── FINSABER/                         # 参考回测框架
│   ├── backtest/
│   │   ├── data_util/
│   │   │   └── finmem_dataset.py     # FinMemDataset 类
│   │   ├── toolkit/
│   │   │   └── backtest_framework_iso.py  # FINSABERFrameworkHelper
│   │   └── strategy/
│   │       └── timing_llm/
│   │           └── base_strategy_iso.py   # BaseStrategyIso
│   ├── rl_traders/                   # RL 智能体实现
│   └── llm_traders/                  # 基于 LLM 的交易策略（FinMem）
│
├── llm_rl_trading_finsaber/          # ★ 重构后的模块化版本
│   ├── src/
│   │   ├── lesr/                     # LESR 核心（llm_sampler, prompt_templates）
│   │   ├── drl/                      # RL 智能体（DQN, TD3, 策略, 回放缓冲区）
│   │   ├── env/                      # 交易环境（Gym, FINSABER 兼容）
│   │   ├── data/                     # 数据加载和特征工程
│   │   ├── llm/                      # LLM 客户端（OpenAI, DeepSeek, 桩）
│   │   ├── pipeline/                 # 编排（分支迭代, 市场状态）
│   │   ├── finsaber_native/          # 原生 FINSABER 集成
│   │   └── utils/                    # 工具（code_loader, hash, paths）
│   ├── configs/                      # YAML 实验配置
│   └── scripts/                      # 运行脚本
│
├── LESR/                             # 原始 LESR 论文代码库
│   ├── models/                       # 原始模型实现
│   └── LESR-resources/               # 论文资源
│
├── data/                             # 数据准备脚本
│   ├── build_dataset_2012_2017.py    # 2012-2017 多股票数据集构建
│   ├── build_sliding_windows.py      # 滑动窗口数据集构建
│   └── preprocess_data.py            # 数据预处理
│
├── yanlin/                           # 合作者早期版本
├── 参考项目梳理/                      # 参考项目分析文档
├── 项目梳理/                          # 项目文档整理
├── 项目分析/                          # 项目分析文档
├── 背景知识学习/                      # 背景知识笔记
└── README.md                         # 项目说明
```

## 关键文件位置

| 功能 | 位置 |
|------|------|
| LESR 优化循环 | `exp4.7/lesr_controller.py` |
| DQN 实现 | `exp4.7/dqn_trainer.py` |
| LLM prompt | `exp4.7/prompts.py` |
| 特征分析 | `exp4.7/feature_analyzer.py` |
| 默认配置 | `exp4.7/config.yaml` |
| 数据集加载器 | `FINSABER/backtest/data_util/finmem_dataset.py` |
| 回测框架 | `FINSABER/backtest/toolkit/backtest_framework_iso.py` |
| 重构版 LESR | `llm_rl_trading_finsaber/src/lesr/` |
| 数据构建脚本 | `data/build_*.py` |

## 命名规范

- 实验目录：`exp{版本}` 或 `{版本}exp`（如 `exp4.7`、`4.12exp`）
- 配置文件：`config.yaml`（默认）、`config_W{N}.yaml`（窗口）、`config_{股票}.yaml`（股票专用）
- 窗口配置：W1-W10 = 不同时间窗口，SW = 滑动窗口
- 结果目录：`result_*` 或 `results_*` 前缀加实验范围
