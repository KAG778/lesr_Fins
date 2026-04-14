# 外部集成

## 外部 API

### OpenAI LLM API（通过 ChatAnywhere 代理）
- **用途**：生成 `revise_state()` 和 `intrinsic_reward()` Python 函数代码
- **模型**：`gpt-4o-mini`
- **端点**：`https://api.chatanywhere.com.cn/v1`（可通过 `llm.base_url` 配置）
- **认证**：配置 YAML 中的 API 密钥 或 `OPENAI_API_KEY` 环境变量
- **调用位置**：`exp4.7/lesr_controller.py` 中的 LESR 优化循环
- **库版本**：`openai==0.28`（旧版 API，非 v1.x）

### FinMemDataset（FINSABER 子模块）
- **用途**：金融数据加载和时间范围筛选
- **位置**：`FINSABER/backtest/data_util/finmem_dataset.py`
- **接口**：`FinMemDataset(pickle_file=)` → `get_subset_by_time_range()`, `get_ticker_price_by_date()`
- **被使用于**：`exp4.7/main.py`、`exp4.7/lesr_controller.py`

### FINSABER 回测框架
- **用途**：策略评估、回测、指标计算
- **位置**：`FINSABER/backtest/toolkit/backtest_framework_iso.py`
- **关键类**：`FINSABERFrameworkHelper`
- **被使用于**：`exp4.7/main.py` 的测试集评估
- **提供**：标准化回测执行、绩效指标（夏普比率、最大回撤、收益率）

## 数据源

### Pickle 数据文件
- **位置**：`data/stock_data_exp4_7.pkl`（及其他）
- **格式**：按日期 → 股票 → OHLCV 价格序列化的字典
- **构建脚本**：`data/build_dataset_2012_2017.py`、`data/build_sliding_windows.py` 等
- **包含**：TSLA、MSFT、AMZN、NFLX 股票；多个时间窗口（2012-2017、滑动窗口）

## LLM 客户端（在 `llm_rl_trading_finsaber/src/llm/` 中）
- `deepseek_client.py` — DeepSeek LLM 支持
- `finagent_stub.py` — FinAgent 集成桩
- `finmem_stub.py` — FinMem 集成桩

## 无数据库
- 没有数据库集成 — 所有数据存储在 pickle 文件中
- 没有缓存层 — 每次运行重新加载数据
