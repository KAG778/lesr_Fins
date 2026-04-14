# 技术栈

## 语言与运行环境

- **Python 3.8+** — 项目唯一编程语言
- **PyTorch** — DQN/TD3 强化学习智能体，神经网络模型，支持 GPU 训练
- **NumPy** — 数组运算，状态表示，特征计算
- **Pandas** — 时间序列数据处理，日期索引

## 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | — | DQN/TD3 神经网络，GPU 训练 |
| numpy | — | 状态数组，特征计算 |
| pandas | — | 金融数据操作 |
| openai | 0.28（旧版） | LLM API 调用 (GPT-4o-mini) |
| scikit-learn | — | 随机森林用于特征重要性分析 |
| shap | — | SHAP 值用于特征分析 |
| pyyaml | — | YAML 配置加载 |
| matplotlib | — | 结果可视化 |
| backtrader | — | 回测引擎 (FINSABER) |
| rich | — | 终端格式化 |
| colorlog | — | 彩色日志 |
| tabulate | — | 表格输出 |
| scipy | — | Spearman 相关性分析 |

## 配置方式

- **YAML 配置文件** — 每个实验一套配置（如 `exp4.7/config.yaml`, `config_W1.yaml` ~ `config_W10.yaml`）
- 配置包含：数据路径、股票代码、训练/验证/测试时间段、LLM 参数、DQN 超参数、内在奖励权重
- API 密钥存储在配置 YAML 或环境变量中（`OPENAI_API_KEY`、`OPENAI_BASE_URL`）
- LLM 端点：ChatAnywhere 代理（`https://api.chatanywhere.com.cn/v1`）

## 数据格式

- Pickle 文件（`.pkl`）— 按日期按股票序列化的每日 OHLCV 数据
- 数据结构：`dict[date, dict]["price"][ticker]`，包含收盘价、开盘价、最高价、最低价、成交量、复权收盘价
- 原始状态：120 维向量（20 天 × 6 特征：收盘价、开盘价、最高价、最低价、成交量、复权收盘价）

## 项目结构模式

多个实验目录（`exp4.7/`、`exp4.9/`、`exp_4.9_b/`、`exp4.9_c/`、`exp4.9_d/`、`EXP4.9_f/`、`4.12exp/`），每个都是 LESR 管道的迭代版本，略有差异。`exp4.7/` 是当前主要活跃版本。
