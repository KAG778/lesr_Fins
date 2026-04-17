# 代码规范

## 代码风格

- **Python 3.8+**，使用类型注解（`typing` 模块）
- 模块级 docstring 用英文，描述模块用途和核心逻辑
- 函数使用 `Args:` / `Returns:` 风格的 docstring
- 日志使用 `logging` 模块，通过 `logger = logging.getLogger(__name__)` 获取

## 导入方式

- 使用 `sys.path.insert(0, ...)` 手动管理模块搜索路径，以项目根目录为基准
- 常见模式：
  ```python
  sys.path.insert(0, str(Path(__file__).parent.parent))           # 上一级目录
  sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))  # FINSABER 子项目
  ```
- 没有使用 `setup.py` 或 `pyproject.toml` 做包管理
- 跨实验目录的导入依赖 `sys.path` hack

## 命名规范

- 类名：大驼峰（`DQNTrainer`、`LESRController`、`FinMemDataset`）
- 函数/变量：蛇形命名（`revise_state`、`intrinsic_reward`、`train_start`）
- 常量：全大写蛇形（`INITIAL_PROMPT`、`MAX_ITERATIONS`）
- 私有函数：单下划线前缀（`_train_ticker_worker`）

## 配置管理

- 所有超参数通过 YAML 配置文件管理
- 配置文件与实验目录绑定（`exp4.7/config.yaml`）
- 不同窗口/股票的配置通过 `config_W{N}.yaml`、`config_{stock}.yaml` 区分
- API 密钥直接写在配置文件中（安全风险 — 见 CONCERNS.md）

## 错误处理

- 使用 `try/except` 包裹外部调用（LLM API、文件 I/O）
- 日志记录错误：`logger.warning()`、`logger.error()`
- 部分函数缺少错误处理（如数据分析函数）
- worker 进程中使用 `try/except` 捕获异常并返回错误信息

## 代码动态执行

- LLM 生成的代码以字符串形式接收
- 通过 `tempfile.NamedTemporaryFile` 写入临时 `.py` 文件
- 使用 `importlib.util.spec_from_file_location` 动态导入
- 临时文件在导入后立即删除（`os.unlink(tmp.name)`）

## 日志规范

- 双输出：文件日志 + 控制台（`FileHandler` + `StreamHandler`）
- 格式：`'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`
- 日志文件存放于各实验的 `logs/` 目录
