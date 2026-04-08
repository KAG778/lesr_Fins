# FINSABER 系统架构分析

## 项目概述

**FINSABER** 是一个综合性的金融交易策略评估框架，专注于比较传统技术分析方法与现代机器学习及大语言模型（LLM）基于策略的性能。该项目已被 KDD 2026 接收。

### 核心目标
- 评估不同交易策略在长期投资中的表现
- 对比传统技术分析、机器学习和 LLM 驱动的策略
- 提供可扩展的回测框架支持自定义策略和数据集

---

## 系统整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        FINSABER 框架                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  数据层      │  │  策略层      │  │  评估层      │          │
│  │  Data Layer  │  │ Strategy     │  │  Evaluation  │          │
│  │              │  │ Layer        │  │  Layer       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           回测引擎 (Backtest Engine)                     │   │
│  │  - FINSABER (LLM策略)  - FINSABERBt (传统策略)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           实验管理 (Experiment Runner)                   │   │
│  │  - 滚动窗口回测  - 多策略对比  - 结果聚合                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心模块详解

### 1. 数据层 (Data Layer)

#### 1.1 数据源
```
数据源
├── 价格数据 (Price Data)
│   ├── 开盘价 (Open)
│   ├── 最高价 (High)
│   ├── 最低价 (Low)
│   ├── 收盘价 (Close)
│   ├── 复权价 (Adjusted Close)
│   └── 成交量 (Volume)
├── 新闻数据 (News Data)
│   └── 新闻标题 (Headlines)
├── 财务文件 (Filings)
│   ├── 10-K 年报
│   └── 10-Q 季报
└── 市场数据
    ├── S&P 500 成分股
    └── 其他指数数据
```

#### 1.2 数据加载器
```python
# 数据加载器层次结构
BacktestDataset (抽象基类)
├── FinmemDataset
│   ├── 加载聚合数据 (价格 + 新闻 + 财务文件)
│   ├── 支持时间范围子集提取
│   └── 支持个股时间序列提取
└── 自定义数据集
    └── 用户可继承扩展
```

**核心接口：**
- `get_ticker_price_by_date(ticker, date)` - 获取特定股票特定日期的价格
- `get_data_by_date(date)` - 获取特定日期的所有数据
- `get_subset_by_time_range(start, end)` - 获取时间范围内的数据子集
- `get_ticker_subset_by_time_range(ticker, start, end)` - 获取特定股票的时间序列
- `get_date_range()` - 获取所有可用日期
- `get_tickers_list()` - 获取所有可交易股票列表

#### 1.3 数据格式
```python
{
    datetime.date(2024, 1, 2): {
        "price": {
            "AAPL": {
                "open": 187.15,
                "high": 188.44,
                "low": 183.89,
                "close": 185.64,
                "adjusted_close": 185.3,
                "volume": 82488700
            },
            # ... 更多股票
        },
        "news": {
            "AAPL": ["headline 1", "headline 2", ...],
            # ... 更多股票
        },
        "filing_k": {
            "AAPL": "10-K filing text...",
            # ... 更多股票
        },
        "filing_q": {
            "AAPL": "10-Q filing text...",
            # ... 更多股票
        }
    },
    # ... 更多日期
}
```

---

### 2. 策略层 (Strategy Layer)

#### 2.1 策略分类体系
```
策略类型
├── 传统技术分析策略 (Timing Strategies)
│   ├── 买入持有策略 (Buy and Hold)
│   ├── 移动平均交叉 (SMA Crossover)
│   ├── 加权移动平均 (WMA Crossover)
│   ├── 布林带策略 (Bollinger Bands)
│   ├── ATR 波动率策略 (ATR Band)
│   ├── 月末效应策略 (Turn of the Month)
│   ├── 趋势跟踪策略 (Trend Following)
│   ├── XGBoost 预测策略
│   └── ARIMA 预测策略
├── LLM 驱动策略 (LLM-based Strategies)
│   ├── FINMEM 策略
│   │   ├── 分层记忆机制
│   │   ├── 角色设计
│   │   └── 多源信息融合
│   └── FINAGENT 策略
│       ├── 强化学习微调
│       └── 环境感知决策
├── 强化学习策略 (RL Strategies)
│   ├── PPO (Proximal Policy Optimization)
│   ├── A2C/A3C (Actor-Critic)
│   ├── SAC (Soft Actor-Critic)
│   └── TD3 (Twin Delayed DDPG)
└── 选择策略 (Selection Strategies)
    ├── FINMEM 选择器
    ├── FINAGENT 选择器
    ├── 随机 S&P 500 选择器
    ├── 动量选择器
    └── 低波动率选择器
```

#### 2.2 传统技术分析策略

**基础策略类：**
```python
class BaseStrategy(bt.Strategy):
    # 统一接口
    params = (
        # 策略参数
    )
    
    def __init__(self):
        # 初始化指标
    
    def next(self):
        # 每个bar的决策逻辑
        # 必须调用 self.post_next_actions()
    
    def post_next_actions(self):
        # 记录状态、计算指标
```

**代表性策略：**

1. **SMA 交叉策略**
```python
class SMACrossStrategy(BaseStrategy):
    params = (
        ("fast_period", 20),
        ("slow_period", 100)
    )
    
    def next(self):
        if self.fast[0] > self.slow[0] and not self.position:
            self.buy()
        elif self.fast[0] < self.slow[0] and self.position:
            self.sell()
```

2. **布林带策略**
```python
class BollingerBandsStrategy(BaseStrategy):
    params = (
        ("period", 20),
        ("stddev", 2)
    )
    
    def next(self):
        if self.data.close[0] < self.lowerband[0]:
            self.buy()  # 价格跌破下轨买入
        elif self.data.close[0] > self.upperband[0]:
            self.sell()  # 价格突破上轨卖出
```

#### 2.3 LLM 驱动策略

**FINMEM 策略架构：**
```
FINMEM 框架
├── 画像模块 (Profiling)
│   ├── 投资者角色设定
│   ├── 风险偏好配置
│   └── 决策风格定义
├── 记忆模块 (Memory)
│   ├── 感知记忆 (Perception Memory)
│   ├── 短期记忆 (Short-term Memory)
│   ├── 长期记忆 (Long-term Memory)
│   └── 记忆检索与更新
└── 决策模块 (Decision-making)
    ├── 信息分析
    ├── 推理链构建
    └── 交易决策生成
```

**核心代码结构：**
```python
class FinMem(BaseStrategyIso):
    def __init__(self, symbol, date_from, date_to, model):
        self.symbol = symbol
        self.model = model  # LLM 客户端
        self.memory = LayeredMemory()
        self.profile = InvestorProfile()
    
    def train(self):
        # 训练阶段：填充记忆
        for date, data in historical_data:
            self.memory.update(date, data)
    
    def on_data(self, date, today_data, framework):
        # 1. 检索相关记忆
        relevant_memories = self.memory.retrieve(date)
        
        # 2. 构建提示词
        prompt = self.build_prompt(date, today_data, relevant_memories)
        
        # 3. LLM 推理
        decision = self.model.decide(prompt)
        
        # 4. 执行交易
        if decision.action == "buy":
            framework.buy(date, self.symbol, price, quantity)
        elif decision.action == "sell":
            framework.sell(date, self.symbol, price, quantity)
        
        # 5. 更新记忆
        self.memory.update(date, today_data, decision)
```

**FINAGENT 策略架构：**
```
FINAGENT 框架
├── 环境交互 (Environment)
│   ├── 市场环境
│   ├── 交易执行
│   └── 奖励计算
├── 强化学习 (RL Fine-tuning)
│   ├── PPO 算法
│   ├── 策略网络
│   └── 价值网络
└── LLM 决策 (LLM Decision)
    ├── 状态编码
    ├── 动作生成
    └── 策略优化
```

#### 2.4 选择策略

**选择策略基类：**
```python
class BaseSelector:
    def select(self, data_loader, start_date, end_date):
        # 返回选定股票列表
        return ["AAPL", "MSFT", "GOOGL", ...]
```

**代表性选择器：**

1. **FINMEM 选择器**
```python
class FinMemSelector(BaseSelector):
    def select(self, data_loader, start_date, end_date):
        # 基于 LLM 分析选择股票
        # 考虑市场趋势、新闻情绪、财务状况
        return selected_tickers
```

2. **动量选择器**
```python
class MomentumSelector(BaseSelector):
    def __init__(self, top_k=5, lookback_period=252):
        self.top_k = top_k
        self.lookback_period = lookback_period
    
    def select(self, data_loader, start_date, end_date):
        # 计算动量指标
        momentum_scores = {}
        for ticker in all_tickers:
            scores = self.calculate_momentum(ticker, start_date, end_date)
            momentum_scores[ticker] = scores
        
        # 选择动量最高的股票
        ranked = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
        return ranked[:self.top_k]
```

---

### 3. 回测引擎层 (Backtest Engine Layer)

#### 3.1 双引擎架构
```
回测引擎
├── FINSABER (LLM 策略引擎)
│   ├── 支持非结构化数据
│   ├── 自然语言处理
│   ├── 复杂决策逻辑
│   └── 迭代式股票回测
└── FINSABERBt (传统策略引擎)
    ├── 基于 Backtrader
    ├── 高性能技术指标计算
    ├── 向量化操作
    └── 单股票回测
```

#### 3.2 FINSABER 引擎 (LLM 专用)
```python
class FINSABER:
    def __init__(self, trade_config):
        self.trade_config = TradeConfig.from_dict(trade_config)
        self.framework = FINSABERFrameworkHelper(
            initial_cash=trade_config.cash,
            risk_free_rate=trade_config.risk_free_rate,
            commission_per_share=trade_config.commission,
            min_commission=trade_config.min_commission
        )
        self.data_loader = trade_config.data_loader
    
    def run_rolling_window(self, strategy_class, rolling_window_size, 
                          rolling_window_step, strat_params):
        """
        滚动窗口回测
        - 支持多时间窗口
        - 自动股票选择
        - 迭代式回测
        """
        rolling_windows = self._generate_rolling_windows(
            rolling_window_size, rolling_window_step
        )
        
        eval_metrics = {}
        for window in rolling_windows:
            # 1. 选择股票
            tickers = self.stock_selector.select(
                self.data_loader, window[0], window[1]
            )
            
            # 2. 运行回测
            for ticker in tickers:
                metrics = self.run_single_ticker(
                    strategy_class, ticker, window, strat_params
                )
                eval_metrics[f"{ticker}_{window[0]}"] = metrics
        
        return eval_metrics
    
    def run_iterative_tickers(self, strategy_class, strat_params, 
                             tickers, delist_check=True):
        """
        迭代式回测多只股票
        """
        metrics = {}
        for ticker in tickers:
            # 初始化策略
            strategy = strategy_class(**strat_params)
            
            # 训练阶段
            strategy.train()
            
            # 测试阶段
            ticker_metrics = self._run_backtest(strategy, ticker)
            metrics[ticker] = ticker_metrics
        
        return metrics
```

#### 3.3 FINSABERBt 引擎 (传统策略专用)
```python
class FINSABERBt:
    def __init__(self, trade_config):
        self.trade_config = trade_config
        # 基于 Backtrader 的 Cerebro 引擎
        self.cerebro = bt.Cerebro()
    
    def run_rolling_window(self, strategy_class, rolling_window_size,
                          rolling_window_step):
        """
        滚动窗口回测 - 传统策略版本
        """
        rolling_windows = self._generate_rolling_windows(
            rolling_window_size, rolling_window_step
        )
        
        eval_metrics = {}
        for window in rolling_windows:
            for ticker in self.trade_config.tickers:
                # 1. 加载数据
                data = self._load_data(ticker, window[0], window[1])
                self.cerebro.adddata(data)
                
                # 2. 添加策略
                self.cerebro.addstrategy(strategy_class)
                
                # 3. 运行回测
                results = self.cerebro.run()
                
                # 4. 提取指标
                metrics = self._extract_metrics(results)
                eval_metrics[f"{ticker}_{window[0]}"] = metrics
        
        return eval_metrics
```

#### 3.4 框架辅助工具
```python
class FINSABERFrameworkHelper:
    """
    交易框架辅助类
    - 管理投资组合状态
    - 执行买卖操作
    - 计算性能指标
    """
    def __init__(self, initial_cash, risk_free_rate, 
                 commission_per_share, min_commission):
        self.cash = initial_cash
        self.portfolio = {}  # {ticker: {quantity, avg_cost}}
        self.risk_free_rate = risk_free_rate
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        
        # 性能跟踪
        self.equity_curve = []
        self.trades_history = []
        self.drawdowns = []
    
    def buy(self, date, ticker, price, quantity):
        """执行买入"""
        cost = price * quantity
        commission = max(cost * self.commission_per_share, self.min_commission)
        total_cost = cost + commission
        
        if total_cost > self.cash:
            raise InsufficientFundsError()
        
        self.cash -= total_cost
        
        if ticker not in self.portfolio:
            self.portfolio[ticker] = {"quantity": 0, "avg_cost": 0}
        
        # 更新持仓
        old_quantity = self.portfolio[ticker]["quantity"]
        old_cost = self.portfolio[ticker]["avg_cost"] * old_quantity
        new_quantity = old_quantity + quantity
        new_avg_cost = (old_cost + cost) / new_quantity
        
        self.portfolio[ticker] = {
            "quantity": new_quantity,
            "avg_cost": new_avg_cost
        }
        
        # 记录交易
        self.trades_history.append({
            "date": date,
            "action": "buy",
            "ticker": ticker,
            "price": price,
            "quantity": quantity,
            "commission": commission
        })
    
    def sell(self, date, ticker, price, quantity):
        """执行卖出"""
        if ticker not in self.portfolio:
            raise InsufficientPositionError()
        
        if quantity > self.portfolio[ticker]["quantity"]:
            raise InsufficientPositionError()
        
        proceeds = price * quantity
        commission = max(proceeds * self.commission_per_share, self.min_commission)
        net_proceeds = proceeds - commission
        
        # 更新持仓
        self.portfolio[ticker]["quantity"] -= quantity
        if self.portfolio[ticker]["quantity"] == 0:
            del self.portfolio[ticker]
        
        self.cash += net_proceeds
        
        # 记录交易
        self.trades_history.append({
            "date": date,
            "action": "sell",
            "ticker": ticker,
            "price": price,
            "quantity": quantity,
            "commission": commission
        })
    
    def calculate_metrics(self):
        """计算性能指标"""
        equity_values = [e["total_equity"] for e in self.equity_curve]
        
        metrics = {
            "total_return": (equity_values[-1] - equity_values[0]) / equity_values[0],
            "annual_return": self._calculate_annual_return(equity_values),
            "annual_volatility": self._calculate_volatility(equity_values),
            "sharpe_ratio": self._calculate_sharpe_ratio(equity_values),
            "sortino_ratio": self._calculate_sortino_ratio(equity_values),
            "max_drawdown": self._calculate_max_drawdown(equity_values),
            "total_commission": sum(t["commission"] for t in self.trades_history)
        }
        
        return metrics
```

---

### 4. 实验管理层 (Experiment Management Layer)

#### 4.1 实验运行器
```python
# 传统策略实验运行器
def run_baselines_exp():
    """
    运行传统技术分析策略实验
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=str, 
                       choices=["cherry_pick_both_finmem", 
                               "cherry_pick_both_fincon",
                               "selected_4",
                               "random_sp500_5",
                               "momentum_sp500_5",
                               "lowvol_sp500_5"])
    parser.add_argument("--include", type=str, 
                       help="策略类名")
    parser.add_argument("--date_from", type=str)
    parser.add_argument("--date_to", type=str)
    parser.add_argument("--training_years", type=int)
    parser.add_argument("--rolling_window_size", type=int)
    parser.add_argument("--rolling_window_step", type=int)
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_experiment(
        setup=args.setup,
        strategy=args.include,
        date_from=args.date_from,
        date_to=args.date_to,
        rolling_window_size=args.rolling_window_size
    )
    
    return results

# LLM 策略实验运行器
def run_llm_traders_exp():
    """
    运行 LLM 驱动策略实验
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=str)
    parser.add_argument("--strategy", type=str, 
                       choices=["FinMem", "FinAgent"])
    parser.add_argument("--strat_config_path", type=str)
    parser.add_argument("--date_from", type=str)
    parser.add_argument("--date_to", type=str)
    parser.add_argument("--rolling_window_size", type=int)
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.strat_config_path, 'r') as f:
        strat_config = json.load(f)
    
    # 运行实验
    results = run_llm_experiment(
        setup=args.setup,
        strategy=args.strategy,
        config=strat_config,
        date_from=args.date_from,
        date_to=args.date_to
    )
    
    return results
```

#### 4.2 实验配置

**实验设置类型：**
```python
实验设置
├── cherry_pick_both_finmem (精选 FINMEM)
│   ├── 手工挑选的最佳表现股票
│   ├── TSLA, AMZN, MSFT, NFLX, COIN
│   └── 用于验证 LLM 策略上限
├── cherry_pick_both_fincon (精选 FINCON)
│   ├── 基于 FINCON 框架的精选股票
│   └── 用于对比不同 LLM 框架
├── selected_4 (选定4股)
│   ├── 固定4只股票组合
│   └── 稳定性测试
├── random_sp500_5 (随机S&P 500)
│   ├── 随机选择5只S&P 500股票
│   └── 泛化能力测试
├── momentum_sp500_5 (动量S&P 500)
│   ├── 基于动量因子选择
│   └── 因子投资验证
└── lowvol_sp500_5 (低波动率S&P 500)
    ├── 基于低波动率因子选择
    └── 风险管理验证
```

**配置文件示例：**
```json
{
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "memory_config": {
        "perception_memory_size": 10,
        "short_term_memory_size": 50,
        "long_term_memory_size": 200
    },
    "profile_config": {
        "risk_tolerance": "moderate",
        "investment_horizon": "medium_term",
        "trading_style": "momentum"
    },
    "trading_config": {
        "initial_cash": 100000,
        "commission_per_share": 0.0049,
        "min_commission": 0.99
    }
}
```

#### 4.3 滚动窗口机制
```
时间轴: 2004 ───── 2006 ───── 2008 ───── 2010 ───── 2012 ───── 2014
        │           │           │           │           │           │
窗口1:   [====训练======测试==]
窗口2:                 [====训练======测试==]
窗口3:                             [====训练======测试==]

参数:
- rolling_window_size: 2年 (训练 + 测试)
- rolling_window_step: 1年 (窗口移动步长)
- training_years: 2年 (训练期长度)
```

**实现逻辑：**
```python
def generate_rolling_windows(date_from, date_to, window_size, step):
    """生成滚动窗口"""
    start_year = pd.to_datetime(date_from).year
    end_year = pd.to_datetime(date_to).year
    total_years = end_year - start_year + 1
    
    windows = []
    for i in range(0, total_years - window_size, step):
        window_start = f"{start_year + i}-01-01"
        window_end = f"{start_year + i + window_size}-01-01"
        windows.append((window_start, window_end))
    
    return windows
```

---

### 5. 性能评估层 (Performance Evaluation Layer)

#### 5.1 评估指标体系
```
性能指标
├── 收益指标
│   ├── 总收益率 (Total Return)
│   ├── 年化收益率 (Annual Return)
│   └── 累积收益率 (Cumulative Return)
├── 风险指标
│   ├── 年化波动率 (Annual Volatility)
│   ├── 最大回撤 (Max Drawdown)
│   └── 下行风险 (Downside Risk)
├── 风险调整收益指标
│   ├── 夏普比率 (Sharpe Ratio)
│   ├── 索提诺比率 (Sortino Ratio)
│   └── 信息比率 (Information Ratio)
└── 交易指标
    ├── 总交易次数 (Total Trades)
    ├── 胜率 (Win Rate)
    ├── 盈亏比 (Profit/Loss Ratio)
    └── 总交易成本 (Total Commission)
```

#### 5.2 指标计算实现
```python
def calculate_performance_metrics(equity_curve, trades_history, risk_free_rate):
    """
    计算完整的性能指标
    """
    # 1. 基础指标
    initial_equity = equity_curve[0]
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_equity) / initial_equity
    
    # 2. 年化收益率
    days = len(equity_curve)
    years = days / 252  # 假设每年252个交易日
    annual_return = (1 + total_return) ** (1 / years) - 1
    
    # 3. 波动率
    returns = pd.Series(equity_curve).pct_change().dropna()
    annual_volatility = returns.std() * np.sqrt(252)
    
    # 4. 夏普比率
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    # 5. 最大回撤
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    # 6. 索提诺比率
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std
    
    # 7. 交易统计
    total_trades = len(trades_history)
    winning_trades = [t for t in trades_history if t["pnl"] > 0]
    losing_trades = [t for t in trades_history if t["pnl"] < 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([abs(t["pnl"]) for t in losing_trades]) if losing_trades else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    total_commission = sum(t["commission"] for t in trades_history)
    
    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "total_commission": total_commission
    }
    
    return metrics
```

#### 5.3 结果聚合与可视化
```python
def aggregate_results(results_dir):
    """
    聚合多个实验的结果
    """
    # 1. 加载所有结果文件
    results = {}
    for setup in os.listdir(results_dir):
        setup_path = os.path.join(results_dir, setup)
        for strategy in os.listdir(setup_path):
            strategy_path = os.path.join(setup_path, strategy)
            with open(os.path.join(strategy_path, "results.csv"), 'r') as f:
                results[f"{setup}_{strategy}"] = pd.read_csv(f)
    
    # 2. 计算汇总统计
    summary = {}
    for key, df in results.items():
        summary[key] = {
            "mean_sharpe": df["sharpe_ratio"].mean(),
            "mean_return": df["annual_return"].mean(),
            "mean_drawdown": df["max_drawdown"].mean(),
            "std_sharpe": df["sharpe_ratio"].std(),
            "std_return": df["annual_return"].std()
        }
    
    # 3. 生成对比图表
    plot_performance_comparison(summary)
    plot_sharpe_heatmap(results)
    
    return summary

def plot_performance_comparison(summary):
    """绘制性能对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 夏普比率对比
    strategies = list(summary.keys())
    sharpes = [summary[s]["mean_sharpe"] for s in strategies]
    axes[0, 0].bar(strategies, sharpes)
    axes[0, 0].set_title("Sharpe Ratio Comparison")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 年化收益对比
    returns = [summary[s]["mean_return"] for s in strategies]
    axes[0, 1].bar(strategies, returns)
    axes[0, 1].set_title("Annual Return Comparison")
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 最大回撤对比
    drawdowns = [summary[s]["mean_drawdown"] for s in strategies]
    axes[1, 0].bar(strategies, drawdowns)
    axes[1, 0].set_title("Max Drawdown Comparison")
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 收益-风险散点图
    axes[1, 1].scatter(returns, sharpes)
    for i, s in enumerate(strategies):
        axes[1, 1].annotate(s, (returns[i], sharpes[i]))
    axes[1, 1].set_xlabel("Annual Return")
    axes[1, 1].set_ylabel("Sharpe Ratio")
    axes[1, 1].set_title("Return-Risk Profile")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("performance_comparison.png", dpi=300)
```

---

## 6. 数据流与执行流程

### 6.1 完整回测流程
```
┌─────────────────────────────────────────────────────────────┐
│                    1. 初始化阶段                             │
├─────────────────────────────────────────────────────────────┤
│  - 加载配置文件 (trade_config)                               │
│  - 初始化数据加载器 (data_loader)                            │
│  - 设置回测参数 (date_from, date_to, cash, commission)      │
│  - 选择股票池 (selection_strategy)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    2. 滚动窗口循环                           │
├─────────────────────────────────────────────────────────────┤
│  for window in rolling_windows:                             │
│    ┌─────────────────────────────────────────────────────┐  │
│    │  2.1 股票选择                                        │  │
│    │  - 执行选择策略 (selection_strategy)                 │  │
│    │  - 获取窗口期股票列表                                │  │
│    └─────────────────────────────────────────────────────┘  │
│                         ↓                                    │
│    ┌─────────────────────────────────────────────────────┐  │
│    │  2.2 迭代回测 (逐股票)                               │  │
│    │  for ticker in tickers:                             │  │
│    │    ┌───────────────────────────────────────────────┐│  │
│    │    │  2.2.1 数据准备                              ││  │
│    │    │  - 加载价格数据                              ││  │
│    │    │  - 加载新闻数据                              ││  │
│    │    │  - 加载财务数据                              ││  │
│    │    └───────────────────────────────────────────────┘│  │
│    │                         ↓                            ││  │
│    │    ┌───────────────────────────────────────────────┐│  │
│    │    │  2.2.2 策略初始化                            ││  │
│    │    │  - 创建策略实例                              ││  │
│    │    │  - 传入配置参数                              ││  │
│    │    └───────────────────────────────────────────────┘│  │
│    │                         ↓                            ││  │
│    │    ┌───────────────────────────────────────────────┐│  │
│    │    │  2.2.3 训练阶段 (LLM策略)                    ││  │
│    │    │  - 历史数据喂入                              ││  │
│    │    │  - 构建记忆                                  ││  │
│    │    │  - 模型微调 (可选)                           ││  │
│    │    └───────────────────────────────────────────────┘│  │
│    │                         ↓                            ││  │
│    │    ┌───────────────────────────────────────────────┐│  │
│    │    │  2.2.4 测试阶段                              ││  │
│    │    │  for date in test_period:                    ││  │
│    │    │    - 获取当日数据                            ││  │
│    │    │    - 策略决策 (buy/sell/hold)               ││  │
│    │    │    - 执行交易                                ││  │
│    │    │    - 更新状态                                ││  │
│    │    │    - 记录指标                                ││  │
│    │    └───────────────────────────────────────────────┘│  │
│    │                         ↓                            ││  │
│    │    ┌───────────────────────────────────────────────┐│  │
│    │    │  2.2.5 计算性能指标                          ││  │
│    │    │  - 总收益率                                  ││  │
│    │    │  - 夏普比率                                  ││  │
│    │    │  - 最大回撤                                  ││  │
│    │    │  - 其他指标                                  ││  │
│    │    └───────────────────────────────────────────────┘│  │
│    └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    3. 结果聚合                               │
├─────────────────────────────────────────────────────────────┤
│  - 汇总所有窗口和股票的结果                                  │
│  - 计算平均指标                                              │
│  - 生成对比图表                                              │
│  - 保存结果文件                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 LLM 策略决策流程
```
┌─────────────────────────────────────────────────────────────┐
│                  FINMEM 决策流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 数据输入                                                 │
│     ├─ 价格数据 (OHLCV)                                     │
│     ├─ 新闻标题 (Headlines)                                 │
│     └─ 财务文件 (10-K, 10-Q)                                │
│                                                              │
│  2. 特征提取                                                 │
│     ├─ 技术指标 (SMA, RSI, MACD)                            │
│     ├─ 价格动量 (Momentum)                                  │
│     ├─ 波动率 (Volatility)                                  │
│     └─ 文本特征 (情感分析, 关键词提取)                       │
│                                                              │
│  3. 记忆检索                                                 │
│     ├─ 感知记忆 (近期市场状态)                              │
│     ├─ 短期记忆 (近期交易决策)                              │
│     ├─ 长期记忆 (历史成功/失败模式)                         │
│     └─ 相关性排序 (Relevance Ranking)                       │
│                                                              │
│  4. 提示词构建                                               │
│     ├─ 系统角色 (System Prompt)                             │
│     │  └─ "你是一位经验丰富的量化交易员..."                 │
│     ├─ 市场背景 (Market Context)                            │
│     │  └─ 当前市场状态、近期走势                           │
│     ├─ 历史记忆 (Historical Memories)                       │
│     │  └─ 相关的过去决策和结果                             │
│     ├─ 当前信息 (Current Information)                       │
│     │  └─ 最新价格、新闻、财务数据                         │
│     └─ 决策要求 (Decision Request)                          │
│        └─ "基于以上信息，请决定: BUY/SELL/HOLD"             │
│                                                              │
│  5. LLM 推理                                                 │
│     ├─ 输入: 完整提示词                                      │
│     ├─ 模型: GPT-4 / Claude / LLaMA                         │
│     ├─ 输出: 结构化决策                                     │
│     │  ├─ 动作: BUY/SELL/HOLD                              │
│     │  ├─ 理由: 自然语言解释                               │
│     │  ├─ 置信度: 0-1 分数                                 │
│     │  └─ 数量: 交易股数                                   │
│     └─ 成本跟踪 (API 调用成本)                              │
│                                                              │
│  6. 决策执行                                                 │
│     ├─ 风险检查                                             │
│     │  ├─ 资金充足性                                       │
│     │  ├─ 持仓限制                                         │
│     │  └─ 风险暴露                                         │
│     ├─ 交易执行                                             │
│     │  ├─ 买入/卖出操作                                    │
│     │  └─ 计算交易成本                                     │
│     └─ 状态更新                                             │
│        ├─ 现金余额                                         │
│        ├─ 持仓数量                                         │
│        └─ 资产总值                                         │
│                                                              │
│  7. 记忆更新                                                 │
│     ├─ 存储决策记录                                         │
│     ├─ 存储市场状态                                         │
│     ├─ 存储执行结果                                         │
│     └─ 更新记忆重要性                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 扩展性与自定义

### 7.1 添加自定义策略

**步骤 1: 创建策略类**
```python
# backtest/strategy/timing/my_custom_strategy.py
import backtrader as bt
from backtest.strategy.timing.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    """
    自定义策略示例
    """
    params = (
        ("param1", 20),
        ("param2", 50),
    )
    
    def __init__(self):
        super().__init__()
        # 初始化指标
        self.indicator1 = bt.indicators.SMA(self.data.close, 
                                            period=self.params.param1)
        self.indicator2 = bt.indicators.RSI(self.data.close, 
                                            period=self.params.param2)
    
    def next(self):
        """
        每个bar的决策逻辑
        """
        # 只在有足够数据时交易
        if len(self.data) < self.params.param2:
            self.post_next_actions()
            return
        
        # 买入条件
        if self.indicator1[0] > self.data.close[0] and \
           self.indicator2[0] < 30 and \
           not self.position:
            size = self._adjust_size_for_commission(
                int(self.broker.cash / self.data.close[0])
            )
            if size > 0:
                self.buy(size=size)
        
        # 卖出条件
        elif self.indicator1[0] < self.data.close[0] and \
             self.position:
            self.close()
        
        # 重要：调用此方法以记录状态
        self.post_next_actions()
```

**步骤 2: 注册策略**
```python
# backtest/strategy/timing/__init__.py
from .my_custom_strategy import MyCustomStrategy

__all__ = [
    # ... 其他策略
    "MyCustomStrategy"
]
```

**步骤 3: 运行自定义策略**
```bash
python backtest/run_baselines_exp.py \
    --setup selected_4 \
    --include MyCustomStrategy \
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --rolling_window_size 2 \
    --rolling_window_step 1
```

### 7.2 添加自定义 LLM 策略

**步骤 1: 创建 LLM 策略类**
```python
# backtest/strategy/timing_llm/my_llm_strategy.py
from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso

class MyLLMStrategy(BaseStrategyIso):
    """
    自定义 LLM 策略示例
    """
    def __init__(self, symbol, date_from, date_to, model, config):
        super().__init__()
        self.symbol = symbol
        self.model = model  # LLM 客户端
        self.config = config
        self.date_from = date_from
        self.date_to = date_to
        
        # 初始化记忆、知识库等
        self.memory = []
        self.knowledge_base = {}
    
    def train(self):
        """
        训练阶段：构建记忆和知识库
        """
        # 获取训练数据
        training_data = self.data_loader.get_subset_by_time_range(
            self.date_from, self.date_to
        )
        
        # 构建记忆
        for date in training_data.get_date_range():
            data = training_data.get_data_by_date(date)
            self._update_memory(date, data)
        
        # 构建知识库
        self._build_knowledge_base()
    
    def on_data(self, date, today_data, framework):
        """
        每个交易日的决策逻辑
        """
        # 1. 提取特征
        features = self._extract_features(today_data)
        
        # 2. 检索相关记忆
        relevant_memories = self._retrieve_memories(date, features)
        
        # 3. 构建提示词
        prompt = self._build_prompt(date, features, relevant_memories)
        
        # 4. LLM 推理
        decision = self.model.decide(prompt)
        
        # 5. 执行交易
        current_price = today_data["price"][self.symbol]["adjusted_close"]
        
        if decision["action"] == "buy":
            # 计算买入数量
            quantity = self._calculate_quantity(
                framework.cash, 
                current_price, 
                decision.get("confidence", 0.5)
            )
            if quantity > 0:
                framework.buy(date, self.symbol, current_price, quantity)
        
        elif decision["action"] == "sell":
            # 卖出全部持仓
            if self.symbol in framework.portfolio:
                quantity = framework.portfolio[self.symbol]["quantity"]
                framework.sell(date, self.symbol, current_price, quantity)
        
        # 6. 更新记忆
        self._update_memory(date, today_data, decision)
    
    def _extract_features(self, data):
        """提取市场特征"""
        price_data = data["price"][self.symbol]
        
        features = {
            "price": price_data["adjusted_close"],
            "volume": price_data["volume"],
            "sma_20": self._calculate_sma(data, 20),
            "rsi": self._calculate_rsi(data, 14),
            # ... 更多特征
        }
        
        # 文本特征
        if "news" in data and self.symbol in data["news"]:
            features["news_sentiment"] = self._analyze_sentiment(
                data["news"][self.symbol]
            )
        
        return features
    
    def _retrieve_memories(self, date, features):
        """检索相关记忆"""
        # 实现记忆检索逻辑
        # 可以基于相似度、时间等
        pass
    
    def _build_prompt(self, date, features, memories):
        """构建 LLM 提示词"""
        prompt = f"""
你是一位专业的量化交易员。请根据以下信息做出交易决策。

【当前市场状态】
日期: {date}
股票: {self.symbol}
当前价格: {features['price']}
RSI: {features['rsi']:.2f}
SMA 20日: {features['sma_20']:.2f}
新闻情绪: {features.get('news_sentiment', 'N/A')}

【历史经验】
{self._format_memories(memories)}

【决策要求】
请基于以上信息，决定是否交易。输出格式：
{{
    "action": "BUY/SELL/HOLD",
    "reason": "决策理由",
    "confidence": 0.0-1.0
}}
"""
        return prompt
```

**步骤 2: 创建配置文件**
```json
// strats_configs/my_llm_strategy_config.json
{
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
    "memory_config": {
        "short_term_size": 30,
        "long_term_size": 100
    },
    "trading_config": {
        "max_position_pct": 0.3,
        "confidence_threshold": 0.6
    }
}
```

**步骤 3: 运行自定义 LLM 策略**
```bash
python backtest/run_llm_traders_exp.py \
    --setup selected_4 \
    --strategy MyLLMStrategy \
    --strat_config_path strats_configs/my_llm_strategy_config.json \
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --rolling_window_size 2
```

### 7.3 添加自定义数据集

**步骤 1: 创建数据集类**
```python
# backtest/data_util/my_custom_dataset.py
from backtest.data_util.backtest_dataset import BacktestDataset

class MyCustomDataset(BacktestDataset):
    """
    自定义数据集示例
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self._load_data()
    
    def _load_data(self):
        """
        加载数据
        """
        # 实现数据加载逻辑
        # 可以从 CSV、数据库、API 等加载
        import pickle
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def get_ticker_price_by_date(self, ticker, date):
        """
        获取特定股票特定日期的价格
        """
        if date not in self.data:
            return None
        if ticker not in self.data[date]["price"]:
            return None
        return self.data[date]["price"][ticker]
    
    def get_data_by_date(self, date):
        """
        获取特定日期的所有数据
        """
        return self.data.get(date, {})
    
    def get_subset_by_time_range(self, start_date, end_date):
        """
        获取时间范围内的数据子集
        """
        subset = {}
        for date, data in self.data.items():
            if start_date <= date <= end_date:
                subset[date] = data
        
        if not subset:
            return None
        
        return MyCustomDataset.__new__(MyCustomDataset, subset)
    
    def get_ticker_subset_by_time_range(self, ticker, start_date, end_date):
        """
        获取特定股票的时间序列
        """
        subset = {}
        for date, data in self.data.items():
            if start_date <= date <= end_date:
                if ticker in data.get("price", {}):
                    subset[date] = {
                        "price": {ticker: data["price"][ticker]}
                    }
        
        if not subset:
            return None
        
        return MyCustomDataset.__new__(MyCustomDataset, subset)
    
    def get_date_range(self):
        """
        获取所有可用日期
        """
        return sorted(self.data.keys())
    
    def get_tickers_list(self):
        """
        获取所有可交易股票
        """
        tickers = set()
        for date_data in self.data.values():
            if "price" in date_data:
                tickers.update(date_data["price"].keys())
        return sorted(tickers)
```

**步骤 2: 使用自定义数据集**
```python
from backtest.finsaber import FINSABER
from backtest.data_util.my_custom_dataset import MyCustomDataset

# 创建数据加载器
data_loader = MyCustomDataset("path/to/my/data.pkl")

# 创建交易配置
trade_config = {
    "tickers": "all",  # 让选择策略决定
    "setup_name": "my_custom_experiment",
    "date_from": "2015-01-01",
    "date_to": "2020-01-01",
    "cash": 100000,
    "commission": 0.0049,
    "min_commission": 0.99,
    "risk_free_rate": 0.02,
    "data_loader": data_loader,
    "rolling_window_size": 2,
    "rolling_window_step": 1,
    "save_results": True,
    "log_base_dir": "results"
}

# 创建引擎并运行
engine = FINSABER(trade_config)
results = engine.run_rolling_window(
    strategy_class=MyLLMStrategy,
    strat_params={
        "symbol": "AAPL",
        "model": my_llm_client,
        "config": my_config
    }
)
```

---

## 8. 项目特点与优势

### 8.1 模块化设计
- **清晰的层次结构**: 数据层、策略层、回测层分离
- **接口统一**: 所有策略实现统一接口
- **易于扩展**: 添加新策略、新数据源无需修改核心代码

### 8.2 多策略支持
- **传统策略**: 技术分析、因子模型
- **机器学习**: XGBoost、ARIMA
- **强化学习**: PPO、A2C、SAC、TD3
- **LLM 策略**: FINMEM、FINAGENT

### 8.3 严谨的回测机制
- **滚动窗口**: 防止前瞻偏差
- **交易成本**: 真实的佣金和滑点模拟
- **风险控制**: 最大回撤、波动率等风险指标
- **性能评估**: 夏普比率、索提诺比率等专业指标

### 8.4 大规模数据处理
- **高效数据结构**: 优化的数据存储和检索
- **并行处理**: 支持多进程回测
- **内存管理**: 智能的数据加载和缓存

### 8.5 实验可复现性
- **配置文件**: 所有参数通过配置文件管理
- **结果保存**: 自动保存中间结果和最终指标
- **日志记录**: 详细的运行日志和错误追踪

### 8.6 LLM 成本追踪
```python
class LLMCostMonitor:
    """
    LLM API 调用成本监控
    """
    def __init__(self):
        self.total_cost = 0.0
        self.call_count = 0
        self.token_count = 0
    
    def record_call(self, model, input_tokens, output_tokens):
        """
        记录 API 调用
        """
        # GPT-4 定价 (示例)
        pricing = {
            "gpt-4": {
                "input": 0.03 / 1000,  # $0.03 per 1K tokens
                "output": 0.06 / 1000
            },
            "gpt-3.5-turbo": {
                "input": 0.0015 / 1000,
                "output": 0.002 / 1000
            }
        }
        
        if model in pricing:
            cost = (input_tokens * pricing[model]["input"] + 
                   output_tokens * pricing[model]["output"])
            self.total_cost += cost
            self.call_count += 1
            self.token_count += input_tokens + output_tokens
    
    def get_summary(self):
        """
        获取成本摘要
        """
        return {
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "token_count": self.token_count,
            "avg_cost_per_call": self.total_cost / self.call_count if self.call_count > 0 else 0
        }
```

---

## 9. 应用场景

### 9.1 学术研究
- **策略对比**: 比较不同类型策略的性能
- **因子研究**: 研究市场因子的有效性
- **LLM 应用**: 探索 LLM 在金融决策中的应用

### 9.2 量化交易
- **策略开发**: 快速开发和测试新策略
- **回测验证**: 历史数据回测验证策略
- **风险管理**: 评估策略的风险收益特征

### 9.3 金融教育
- **教学工具**: 用于金融工程、量化投资教学
- **案例分析**: 分析经典交易策略
- **实训练习**: 提供真实的回测环境

### 9.4 投资实践
- **策略优化**: 优化现有交易策略
- **资产配置**: 辅助投资组合配置决策
- **风险评估**: 评估投资策略的风险

---

## 10. 总结

FINSABER 是一个功能强大、设计优雅的金融交易策略评估框架。它通过模块化的架构设计，支持从传统技术分析到前沿 LLM 驱动策略的全方位评估。

### 核心优势
1. **全面性**: 覆盖传统 ML、RL、LLM 多种策略类型
2. **严谨性**: 严格的回测机制，防止常见偏差
3. **扩展性**: 易于添加自定义策略和数据源
4. **实用性**: 提供完整的实验管理和结果分析工具

### 适用人群
- **研究人员**: 金融、AI 领域的研究者
- **量化从业者**: 量化交易员、策略开发者
- **教育工作者**: 金融、计算机专业教师
- **学生**: 金融工程、计算机专业学生

FINSABER 为金融科技领域的研究和实践提供了一个强大的平台，推动了 LLM 在金融决策中的应用探索。
