# FINSABER 提示词模板代码实现

## 一、提示词模板生成器

### 1.1 FINSABER 特定的提示词生成器

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class FinancialTaskType(Enum):
    """金融任务类型"""
    TIMING = "timing"           # 择时策略
    SELECTION = "selection"     # 选股策略
    ALLOCATION = "allocation"   # 仓位配置

class PerformanceMetric(Enum):
    """性能指标类型"""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"

@dataclass
class FinancialTaskConfig:
    """金融任务配置"""
    task_name: str
    task_type: FinancialTaskType
    raw_state_dim: int
    price_history_days: int
    available_data: List[str]  # ['close', 'open', 'high', 'low', 'volume']
    performance_metric: PerformanceMetric
    risk_tolerance: str = "moderate"  # 'conservative', 'moderate', 'aggressive'
    market_focus: str = "us_equity"  # 'us_equity', 'crypto', 'forex'
    additional_requirements: str = ""

class FINSABERPromptGenerator:
    """
    FINSABER 提示词生成器

    功能:
    1. 生成初始化提示词
    2. 生成COT反馈提示词
    3. 生成迭代改进提示词
    4. 支持多种金融任务类型
    """

    def __init__(self):
        self.financial_knowledge_base = self._load_financial_knowledge()

    def _load_financial_knowledge(self) -> Dict:
        """
        加载金融领域知识库
        """
        return {
            'technical_indicators': {
                'trend': ['SMA', 'EMA', 'MACD', 'ADX'],
                'momentum': ['RSI', 'Stochastic', 'ROC', 'Williams %R'],
                'volatility': ['Bollinger Bands', 'ATR', 'Keltner Channels'],
                'volume': ['OBV', 'Volume MA', 'Volume Rate of Change', 'Chaikin MF']
            },
            'calculation_formulas': {
                'RSI': """
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss
Average Gain = 平均上涨幅度
Average Loss = 平均下跌幅度
""",
                'MACD': """
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(MACD Line, 9)
Histogram = MACD Line - Signal Line
""",
                'ATR': """
TR = max(High - Low, abs(High - Prev_Close), abs(Low - Prev_Close))
ATR = SMA(TR, 14)
"""
            },
            'risk_principles': [
                "不要把所有资金投入单一交易",
                "设置止损点限制潜在损失",
                "考虑波动率调整仓位大小",
                "避免过度交易"
            ],
            'market_regimes': {
                'bull': "价格持续上涨，市场乐观",
                'bear': "价格持续下跌，市场悲观",
                'sideways': "价格在一定范围内波动"
            }
        }

    def generate_init_prompt(self, config: FinancialTaskConfig) -> str:
        """
        生成初始化提示词

        Args:
            config: 金融任务配置

        Returns:
            str: 初始化提示词模板
        """
        # 1. 构建数据描述
        data_description = self._build_data_description(config)

        # 2. 构建特征建议
        feature_suggestions = self._build_feature_suggestions(config)

        # 3. 构建奖励设计原则
        reward_principles = self._build_reward_principles(config)

        # 4. 组装完整Prompt
        prompt = f"""
你是金融交易领域的量化分析专家和特征工程专家。你的任务是为{config.task_name}生成状态表示函数和内在奖励函数。

## 任务背景

{self._get_task_background(config)}

## 可用原始数据

### 数据说明

{data_description}

### 关键金融概念

**收益率 (Return)**:
价格变化的百分比，是金融分析中最基础的度量。
计算公式: Return = (Price_t - Price_{t-1}) / Price_{t-1}
用途: 比原始价格更平稳，更适合分析

**波动率 (Volatility)**:
价格变化的不确定性，是风险的度量。
计算公式: Volatility = Std(Returns)
用途: 高波动率 = 高风险

**动量 (Momentum)**:
价格变化的速度和方向。
计算公式: Momentum = Price_t - Price_{t-n}
用途: 识别趋势强度

**成交量 (Volume)**:
交易的股票数量，反映市场活跃度。
用途: 确认价格趋势的可靠性

## 任务要求

请生成两个Python函数：

### 函数 1: `revise_state(raw_state)`

**输入**:
- `raw_state`: numpy数组，形状({config.raw_state_dim},)，包含{config.price_history_days}天的原始市场数据

**输出**:
- `enhanced_state`: numpy数组，形状({config.raw_state_dim} + k,)，其中k是你添加的特征数量
  - 前{config.raw_state_dim}维保留原始数据
  - 后k维是你计算的金融特征

**建议的特征类别**:

{feature_suggestions}

### 函数 2: `intrinsic_reward(enhanced_state)`

**输入**:
- `enhanced_state`: revise_state的输出，包含原始数据和计算特征

**输出**:
- `reward_value`: 标量值，范围[-100, 100]
  - 正值表示有利于交易的状态
  - 负值表示不利于交易的状态

**设计原则**:

{reward_principles}

## 约束条件

### 1. 数值稳定性
```python
# 必须处理除零、空值等边界情况
❌ 错误: ratio = price / volume
✅ 正确: ratio = price / (volume + 1e-6)

# 必须处理NaN、Inf
❌ 错误: return np.log(price)
✅ 正确: return np.log(price + 1e-6)
```

### 2. 特征范围
```python
# 建议将特征值限制在合理范围
# 过大的特征值会导致训练不稳定
✅ 正确: feature = np.clip(feature, -10, 10)
```

### 3. 维度限制
```python
# 不要添加过多特征（避免过拟合）
# 建议: 5-15个额外特征
```

### 4. 计算效率
```python
# 使用向量化操作，避免循环
❌ 错误:
for i in range(len(close)):
    for j in range(i+1, len(close)):
        # O(n^2)复杂度

✅ 正确:
# 使用numpy向量化操作
np.dot(close, weights)  # O(n)复杂度
```

## 输出格式

请返回完整可执行的Python代码：

```python
import numpy as np

def revise_state(raw_state):
    \"\"\"
    将原始状态转换为增强状态

    Args:
        raw_state: numpy数组，形状({config.raw_state_dim},)，包含{config.price_history_days}天OHLCV数据

    Returns:
        enhanced_state: numpy数组，形状({config.raw_state_dim} + k,)，包含原始数据和计算特征
    \"\"\"
    # 提取各类数据
    # ...

    # 计算特征
    # ...

    # 返回增强状态
    return np.concatenate([raw_state, [feature1, feature2, ...]])

def intrinsic_reward(enhanced_state):
    \"\"\"
    计算内在奖励

    Args:
        enhanced_state: numpy数组，revise_state的输出

    Returns:
        reward: 标量值，范围[-100, 100]
    \"\"\"
    # 提取特征
    # ...

    # 计算奖励
    reward = ...

    # 确保在范围内
    return np.clip(reward, -100, 100)
```

## 重要提示

1. **金融领域知识**: 利用你对技术分析、量化金融的理解来设计特征
2. **简单有效**: 简单的特征组合往往优于复杂的特征工程
3. **避免过拟合**: 不要添加过多特征，5-15个为宜
4. **数值稳定**: 务必处理边界情况，确保代码健壮性
5. **可解释性**: 优先选择有明确金融含义的特征

{config.additional_requirements}

开始生成代码：
"""
        return prompt.strip()

    def _get_task_background(self, config: FinancialTaskConfig) -> str:
        """
        获取任务背景描述
        """
        if config.task_type == FinancialTaskType.TIMING:
            return """
我们将使用强化学习来训练股票择时策略。策略在每个交易日需要决定：
- 买入 (Buy): 预期价格上涨，建立多头仓位
- 卖出 (Sell): 预期价格下跌，平仓或建立空头仓位
- 持有 (Hold): 保持当前仓位不变

为了帮助强化学习算法更好地学习，我们需要设计合适的状态表示和内在奖励。
择时策略的成功依赖于：
1. 准确识别价格趋势
2. 及时捕捉趋势反转
3. 控制风险和回撤
4. 避免过度交易
"""
        elif config.task_type == FinancialTaskType.SELECTION:
            return """
我们将使用强化学习来训练股票选择策略。策略在每个调仓期需要决定：
- 选择哪些股票买入
- 选择哪些股票卖出
- 如何分配资金

为了帮助强化学习算法更好地学习，我们需要设计合适的状态表示和内在奖励。
选股策略的成功依赖于：
1. 识别具有上涨潜力的股票
2. 分散投资降低风险
3. 动态调整投资组合
4. 控制换手率
"""
        else:
            return """
我们将使用强化学习来训练仓位配置策略。策略需要决定：
- 每个资产的仓位权重
- 风险资产与无风险资产的比例
- 如何在不同市场环境下调整仓位

为了帮助强化学习算法更好地学习，我们需要设计合适的状态表示和内在奖励。
仓位配置的成功依赖于：
1. 最大化风险调整后收益
2. 控制投资组合波动率
3. 适应不同市场环境
4. 保持适度分散
"""

    def _build_data_description(self, config: FinancialTaskConfig) -> str:
        """
        构建数据描述
        """
        description = f"""
原始状态数组格式（{config.raw_state_dim}维）:
"""

        if 'close' in config.available_data:
            description += f"""
- s[0:{config.price_history_days-1}]: 收盘价 (Close Price) - 单位: 美元 ($)
  - 最常用的价格数据，反映资产的最终交易价格
"""

        if 'open' in config.available_data:
            description += f"""
- s[{config.price_history_days}:{config.price_history_days*2-1}]: 开盘价 (Open Price) - 单位: 美元 ($)
  - 每个交易日第一笔交易的价格
"""

        if 'high' in config.available_data:
            description += f"""
- s[{config.price_history_days*2}:{config.price_history_days*3-1}]: 最高价 (High Price) - 单位: 美元 ($)
  - 当日交易的最高价格，反映日内波动上限
"""

        if 'low' in config.available_data:
            description += f"""
- s[{config.price_history_days*3}:{config.price_history_days*4-1}]: 最低价 (Low Price) - 单位: 美元 ($)
  - 当日交易的最低价格，反映日内波动下限
"""

        if 'volume' in config.available_data:
            description += f"""
- s[{config.price_history_days*4}:{config.price_history_days*5-1}]: 成交量 (Volume) - 单位: 股数 (shares)
  - 当日交易的股票数量，反映市场活跃度
"""

        description += f"""

**重要提示**:
- 价格数据是非平稳的（随时间变化），通常使用**收益率**而非原始价格
- 收益率 = (当前价格 - 过去价格) / 过去价格
- 成交量通常需要**对数变换**或**标准化**
- 不同资产的尺度差异很大，需要考虑归一化
"""

        return description

    def _build_feature_suggestions(self, config: FinancialTaskConfig) -> str:
        """
        构建特征建议
        """
        suggestions = f"""
#### 类别 1: 趋势指标 (Trend Indicators)
趋势反映价格的运动方向，是交易决策的基础。

推荐特征:
```python
# 价格动量 (Price Momentum)
momentum_5d = (close[-1] - close[-6]) / close[-6]  # 5日收益率
momentum_10d = (close[-1] - close[-11]) / close[-11]  # 10日收益率
momentum_20d = (close[-1] - close[-21]) / close[-21]  # 20日收益率

# 移动平均线 (Moving Average)
ma5 = np.mean(close[-5:])    # 5日均线
ma10 = np.mean(close[-10:])  # 10日均线
ma20 = np.mean(close[-20:])  # 20日均线

# 趋势强度
trend_strength = (ma5 - ma20) / ma20  # 短期均线相对长期均线
```

#### 类别 2: 动量指标 (Momentum Indicators)
动量反映价格变化的速度和力度，帮助识别超买超卖。

推荐特征:
```python
# RSI (Relative Strength Index) - 相对强弱指标
# RSI 衡量价格变动的速度和变化，范围[0, 100]
# RSI > 70: 超买，价格可能回调
# RSI < 30: 超卖，价格可能反弹
delta = np.diff(close)
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = np.mean(gain[-14:])
avg_loss = np.mean(loss[-14:])
rs = avg_gain / (avg_loss + 1e-6)
rsi = 100 - (100 / (1 + rs))

# 随机指标 (Stochastic Oscillator)
stoch_k = 100 * (close[-1] - low[-14:]) / (high[-14:] - low[-14:] + 1e-6)
```

#### 类别 3: 波动率指标 (Volatility Indicators)
波动率反映价格的不确定性，是风险的重要度量。

推荐特征:
```python
# 历史波动率 (Historical Volatility)
returns = np.diff(close) / close[:-1]
volatility_10d = np.std(returns[-10:])  # 10日波动率
volatility_20d = np.std(returns[-20:])  # 20日波动率

# ATR (Average True Range) - 平均真实波幅
# ATR 考虑了跳空情况，更真实地反映波动
prev_close = np.roll(close, 1)
prev_close[0] = close[0]
tr = np.maximum(high - low,
                np.maximum(abs(high - prev_close),
                          abs(low - prev_close)))
atr = np.mean(tr[-14:])

# 布林带宽度 (Bollinger Bands Width)
# 布林带宽度反映波动率的变化
bb_upper = ma20 + 2 * volatility_20d
bb_lower = ma20 - 2 * volatility_20d
bb_width = (bb_upper - bb_lower) / ma20
```

#### 类别 4: 成交量指标 (Volume Indicators)
成交量反映市场参与度和价格变动的可信度。

推荐特征:
```python
# 成交量变化率
volume_change = (volume[-1] - volume[-6]) / volume[-6]

# 成交量移动平均
volume_ma5 = np.mean(volume[-5:])
volume_ma20 = np.mean(volume[-20:])

# 价量关系 (Price-Volume Relation)
# 成交量放大 + 价格上涨 = 上涨趋势确认
price_change = (close[-1] - close[-2]) / close[-2]
volume_price_trend = price_change * (volume[-1] / volume_ma20)
```
"""
        return suggestions

    def _build_reward_principles(self, config: FinancialTaskConfig) -> str:
        """
        构建奖励设计原则
        """
        if config.task_type == FinancialTaskType.TIMING:
            return """
#### 原则 1: 趋势跟随
```python
# 如果处于上升趋势，给予正奖励
if momentum_5d > 0 and momentum_10d > 0:
    reward = 10.0 * momentum_5d  # 上升趋势，正奖励
elif momentum_5d < 0 and momentum_10d < 0:
    reward = -10.0 * abs(momentum_5d)  # 下降趋势，负奖励
```

#### 原则 2: 风险调整
```python
# 考虑波动率，高风险状态给予惩罚
volatility_penalty = -5.0 * volatility_10d
reward += volatility_penalty
```

#### 原则 3: 极端值惩罚
```python
# RSI极端值（超买超卖）给予惩罚
if rsi > 70 or rsi < 30:
    reward -= 5.0
```

#### 原则 4: 成交量确认
```python
# 成交量放大的趋势更可靠
if volume_price_trend > 0:
    reward += 2.0  # 价量配合，额外奖励
```
"""
        else:
            return """
#### 原则 1: 收益导向
```python
# 奖励正收益状态
reward = 10.0 * expected_return
```

#### 原则 2: 风险惩罚
```python
# 惩罚高风险状态
risk_penalty = -5.0 * volatility
reward += risk_penalty
```

#### 原则 3: 分散化奖励
```python
# 奖励分散投资
diversification_bonus = 2.0 * diversification_ratio
reward += diversification_bonus
```
"""

    def generate_cot_prompt(
        self,
        config: FinancialTaskConfig,
        training_results: List[Dict],
        iteration: int
    ) -> str:
        """
        生成COT反馈提示词

        Args:
            config: 金融任务配置
            training_results: 训练结果列表
            iteration: 当前迭代轮数

        Returns:
            str: COT反馈提示词
        """
        # 1. 分析结果
        analysis = self._analyze_training_results(training_results)

        # 2. 生成反馈
        feedback = f"""
我们已成功使用 {len(training_results)} 个不同的状态表示函数训练了{config.task_name}，每个函数都关联一个策略的回测结果。

在训练过程中，我们监控了:
1. 策略的绩效指标
   - 夏普比率 (Sharpe Ratio): 风险调整后收益
   - 总收益 (Total Return): 累积收益率
   - 最大回撤 (Max Drawdown): 最大损失幅度
   - 胜率 (Win Rate): 盈利交易占比

2. 特征重要性分析
   - 与收益的相关系数
   - 与风险的相关系数
   - 特征预测力排名

以下是详细结果:

{analysis['detailed_results']}

**绩效分析**:
- 最佳策略夏普比率: {analysis['best_sharpe']:.3f} (样本 #{analysis['best_id']})
- 最差策略夏普比率: {analysis['worst_sharpe']:.3f} (样本 #{analysis['worst_id']})
- 平均夏普比率: {analysis['avg_sharpe']:.3f}
- 夏普比率标准差: {analysis['std_sharpe']:.3f}

**特征重要性洞察**:

最佳样本 (#{analysis['best_id']}) 的特征分析:
{analysis['best_feature_analysis']}

成功原因:
{analysis['best_success_reasons']}

最差样本 (#{analysis['worst_id']}) 的特征分析:
{analysis['worst_feature_analysis']}

失败原因:
{analysis['worst_failure_reasons']}

**关键发现**:

1. 高相关特征:
{analysis['high_correlation_features']}

2. 低相关特征:
{analysis['low_correlation_features']}

3. 特征组合模式:
{analysis['feature_combination_patterns']}

**改进建议**:

基于以上分析，请回答以下问题:

(a) 为什么最佳样本的特征组合有效？从金融理论角度分析

(b) 为什么最差样本的特征组合失败？识别关键问题

(c) 如何改进特征设计？
    - 应该保留哪些特征？
    - 应该移除哪些特征？
    - 应该尝试哪些新特征？

(d) 如何优化内在奖励函数？
    - 当前奖励设计的问题
    - 改进方向
    - 权重调整建议

(e) 如何避免过拟合？
    - 特征数量控制
    - 正则化方法
    - 样本外验证

请基于以上分析，提供改进的状态表示函数和内在奖励函数。重点关注:
- 保留并优化高相关性特征
- 简化或移除低相关性特征
- 尝试新的特征组合
- 改进内在奖励函数设计
- 确保泛化能力

开始生成改进的代码：
"""
        return feedback.strip()

    def _analyze_training_results(self, results: List[Dict]) -> Dict:
        """
        分析训练结果
        """
        # 按夏普比率排序
        sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'])
        worst = sorted_results[0]
        best = sorted_results[-1]

        # 计算统计量
        sharpes = [r['sharpe_ratio'] for r in results]
        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)

        # 特征重要性分析
        best_features = self._extract_feature_info(best)
        worst_features = self._extract_feature_info(worst)

        # 生成分析
        analysis = {
            'best_sharpe': best['sharpe_ratio'],
            'worst_sharpe': worst['sharpe_ratio'],
            'avg_sharpe': avg_sharpe,
            'std_sharpe': std_sharpe,
            'best_id': results.index(best),
            'worst_id': results.index(worst),
            'best_feature_analysis': self._format_feature_analysis(best_features),
            'worst_feature_analysis': self._format_feature_analysis(worst_features),
            'best_success_reasons': self._identify_success_reasons(best),
            'worst_failure_reasons': self._identify_failure_reasons(worst),
            'high_correlation_features': self._get_high_corr_features(results),
            'low_correlation_features': self._get_low_corr_features(results),
            'feature_combination_patterns': self._analyze_feature_patterns(results),
            'detailed_results': self._format_detailed_results(results)
        }

        return analysis

    def _extract_feature_info(self, result: Dict) -> Dict:
        """提取特征信息"""
        return result.get('feature_analysis', {})

    def _format_feature_analysis(self, features: Dict) -> str:
        """格式化特征分析"""
        if not features:
            return "无详细分析"

        lines = []
        for feature_name, importance in features.items():
            lines.append(f"- {feature_name}: 重要性 {importance:.3f}")

        return "\n".join(lines)

    def _identify_success_reasons(self, result: Dict) -> str:
        """识别成功原因"""
        reasons = []

        # 基于特征数量
        n_features = len(result.get('features', []))
        if 5 <= n_features <= 15:
            reasons.append("✅ 特征数量适中（5-15个），避免过拟合")
        elif n_features > 20:
            reasons.append("❌ 特征过多（>20个），可能过拟合")
        else:
            reasons.append("⚠️  特征较少（<5个），可能信号单一")

        # 基于特征类型
        features = result.get('features', [])
        if 'momentum' in str(features).lower():
            reasons.append("✅ 使用动量指标捕捉趋势")
        if 'volatility' in str(features).lower():
            reasons.append("✅ 考虑波动率进行风险调整")
        if 'volume' in str(features).lower():
            reasons.append("✅ 使用成交量确认趋势")

        return "\n".join(reasons) if reasons else "无明确原因"

    def _identify_failure_reasons(self, result: Dict) -> str:
        """识别失败原因"""
        reasons = []

        # 基于特征数量
        n_features = len(result.get('features', []))
        if n_features > 20:
            reasons.append("❌ 特征过多导致过拟合")
        elif n_features < 3:
            reasons.append("❌ 特征过少，信号单一")

        # 基于夏普比率
        if result['sharpe_ratio'] < 0.5:
            reasons.append("❌ 夏普比率过低，风险调整后收益差")
        if result['max_drawdown'] > 0.3:
            reasons.append("❌ 最大回撤过大（>30%），风险控制差")

        # 基于特征类型
        features = result.get('features', [])
        if 'price' in str(features).lower() and 'return' not in str(features).lower():
            reasons.append("❌ 使用原始价格而非收益率，非平稳问题")

        return "\n".join(reasons) if reasons else "无明确原因"

    def _get_high_corr_features(self, results: List[Dict]) -> str:
        """获取高相关性特征"""
        # 聚合所有结果的特征重要性
        all_importance = {}
        for result in results:
            for feature, imp in result.get('feature_importance', {}).items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(imp)

        # 计算平均重要性
        avg_importance = {
            k: np.mean(v) for k, v in all_importance.items()
        }

        # 排序
        sorted_features = sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 返回前5个
        top_features = [f"{k}: {v:.3f}" for k, v in sorted_features[:5]]
        return "\n".join(top_features)

    def _get_low_corr_features(self, results: List[Dict]) -> str:
        """获取低相关性特征"""
        # 类似 _get_high_corr_features
        all_importance = {}
        for result in results:
            for feature, imp in result.get('feature_importance', {}).items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(imp)

        avg_importance = {
            k: np.mean(v) for k, v in all_importance.items()
        }

        sorted_features = sorted(
            avg_importance.items(),
            key=lambda x: x[1]
        )

        # 返回后5个
        bottom_features = [f"{k}: {v:.3f}" for k, v in sorted_features[:5]]
        return "\n".join(bottom_features)

    def _analyze_feature_patterns(self, results: List[Dict]) -> str:
        """分析特征组合模式"""
        patterns = []

        # 分析特征类型分布
        feature_types = {
            'trend': 0,
            'momentum': 0,
            'volatility': 0,
            'volume': 0
        }

        for result in results:
            features_str = str(result.get('features', [])).lower()
            if 'momentum' in features_str or 'ma' in features_str:
                feature_types['trend'] += 1
            if 'rsi' in features_str or 'stoch' in features_str:
                feature_types['momentum'] += 1
            if 'volatility' in features_str or 'atr' in features_str or 'bb' in features_str:
                feature_types['volatility'] += 1
            if 'volume' in features_str or 'obv' in features_str:
                feature_types['volume'] += 1

        # 识别常见模式
        if feature_types['trend'] > len(results) * 0.5:
            patterns.append("- 趋势指标常见（动量、均线）")
        if feature_types['volatility'] > len(results) * 0.5:
            patterns.append("- 波动率指标常见（风险调整）")
        if feature_types['volume'] > len(results) * 0.3:
            patterns.append("- 成交量确认较常见")

        return "\n".join(patterns) if patterns else "- 无明显模式"

    def _format_detailed_results(self, results: List[Dict]) -> str:
        """格式化详细结果"""
        lines = []

        for idx, result in enumerate(results):
            lines.append(f"""
========== 状态表示函数 -- {idx + 1} ==========
```python
{result.get('code', '代码未提供')}
```
========== 绩效指标 -- {idx + 1} ==========
- 夏普比率: {result['sharpe_ratio']:.3f}
- 总收益: {result['total_return']:.3f}
- 最大回撤: {result['max_drawdown']:.3f}
- 胜率: {result['win_rate']:.3f}

========== 特征分析 -- {idx + 1} ==========
特征列表: {result.get('features', [])}
特征重要性: {result.get('feature_importance', {})}

======================================================================
""")

        return "\n".join(lines)


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 创建生成器
    generator = FINSABERPromptGenerator()

    # 配置任务
    config = FinancialTaskConfig(
        task_name="股票择时策略",
        task_type=FinancialTaskType.TIMING,
        raw_state_dim=120,  # 20天 × 6个OHLCV字段
        price_history_days=20,
        available_data=['close', 'open', 'high', 'low', 'volume'],
        performance_metric=PerformanceMetric.SHARPE_RATIO,
        risk_tolerance="moderate",
        market_focus="us_equity",
        additional_requirements="""
**特别注意**:
- 避免使用未来数据（look-ahead bias）
- 确保特征计算只使用历史数据
- 考虑交易成本和滑点
- 控制交易频率
"""
    )

    # 生成初始化提示词
    init_prompt = generator.generate_init_prompt(config)
    print("=== 初始化提示词 ===")
    print(init_prompt[:500] + "...")
```

## 二、代码验证工具

### 2.1 金融代码验证器

```python
class FinancialCodeValidator:
    """
    金融代码验证器

    功能:
    1. 验证代码可执行性
    2. 检查金融特定约束
    3. 检测look-ahead bias
    4. 评估数值稳定性
    """

    @staticmethod
    def validate_revise_state(
        code: str,
        test_state: np.ndarray,
        expected_input_dim: int,
        verbose: bool = True
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        验证revise_state函数

        Args:
            code: Python代码
            test_state: 测试状态
            expected_input_dim: 期望的输入维度
            verbose: 是否打印详细信息

        Returns:
            Tuple[bool, Optional[str], Optional[int]]:
                (是否通过, 错误信息, 输出维度)
        """
        try:
            # 1. 提取并执行代码
            # ... (类似LESR的代码提取)

            # 2. 测试revise_state
            revised_state = module.revise_state(test_state)

            # 3. 基础验证
            if revised_state.ndim != 1:
                return False, "输出维度错误", None

            if len(revised_state) <= expected_input_dim:
                return False, "未添加额外维度", None

            # 4. 金融特定验证
            # 检查数值稳定性
            if np.any(np.isnan(revised_state)):
                return False, "包含NaN值", None

            if np.any(np.isinf(revised_state)):
                return False, "包含Inf值", None

            # 检查特征值范围
            added_features = revised_state[expected_input_dim:]
            if np.any(np.abs(added_features) > 100):
                if verbose:
                    print(f"⚠️  警告: 特征值过大，可能导致训练不稳定")
                    print(f"   最大特征值: {np.max(np.abs(added_features)):.2f}")

            # 5. 检查是否使用收益率（而非原始价格）
            code_lower = code.lower()
            if 'price' in code_lower and 'return' not in code_lower:
                if verbose:
                    print(f"⚠️  警告: 可能使用原始价格而非收益率")
                    print(f"   建议: 使用收益率而非原始价格")

            # 6. 检查是否有look-ahead bias
            if 'shift(' in code or 'future' in code_lower:
                if verbose:
                    print(f"❌ 错误: 可能存在look-ahead bias")
                return False, "可能存在look-ahead bias", None

            if verbose:
                print(f"✅ 代码验证通过: {expected_input_dim} → {len(revised_state)} 维")

            return True, None, len(revised_state)

        except Exception as e:
            if verbose:
                print(f"❌ 代码验证失败: {e}")
            return False, str(e), None

    @staticmethod
    def validate_intrinsic_reward(
        code: str,
        test_state: np.ndarray,
        verbose: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        验证intrinsic_reward函数

        Args:
            code: Python代码
            test_state: 测试状态
            verbose: 是否打印详细信息

        Returns:
            Tuple[bool, Optional[str]]: (是否通过, 错误信息)
        """
        try:
            # 1. 执行代码
            # ...

            # 2. 测试intrinsic_reward
            reward = module.intrinsic_reward(test_state)

            # 3. 基础验证
            if not isinstance(reward, (int, float, np.number)):
                return False, f"输出类型错误: {type(reward)}"

            # 4. 范围验证
            if reward < -100 or reward > 100:
                return False, f"超出范围 [-100, 100]: {reward}"

            # 5. 金融特定验证
            # 检查奖励是否考虑风险
            code_lower = code.lower()
            if 'volatility' not in code_lower and 'risk' not in code_lower:
                if verbose:
                    print(f"⚠️  警告: 奖励函数可能未考虑风险")

            # 检查是否有clip
            if 'clip' not in code_lower:
                if verbose:
                    print(f"⚠️  警告: 未使用clip，可能超出范围")

            if verbose:
                print(f"✅ 奖励函数验证通过: reward = {reward:.2f}")

            return True, None

        except Exception as e:
            if verbose:
                print(f"❌ 奖励函数验证失败: {e}")
            return False, str(e)
```

## 三、完整的使用示例

```python
# ========== 完整使用示例 ==========

"""
FINSABER 提示词生成和使用流程

Step 1: 创建生成器
Step 2: 配置任务
Step 3: 生成初始化提示词
Step 4: 调用LLM生成代码
Step 5: 验证生成的代码
Step 6: 运行回测
Step 7: 生成COT反馈
Step 8: 迭代优化
"""

# Step 1: 创建生成器
generator = FINSABERPromptGenerator()

# Step 2: 配置任务
config = FinancialTaskConfig(
    task_name="AAPL股票择时策略",
    task_type=FinancialTaskType.TIMING,
    raw_state_dim=120,
    price_history_days=20,
    available_data=['close', 'open', 'high', 'low', 'volume'],
    performance_metric=PerformanceMetric.SHARPE_RATIO,
    risk_tolerance="moderate",
    market_focus="us_equity"
)

# Step 3: 生成初始化提示词
init_prompt = generator.generate_init_prompt(config)

# Step 4: 调用LLM生成代码
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": init_prompt}],
    temperature=0.0
)
generated_code = response['choices'][0]['message']['content']

# Step 5: 验证生成的代码
validator = FinancialCodeValidator()
test_state = np.random.rand(120)

is_valid, error_msg, output_dim = validator.validate_revise_state(
    generated_code,
    test_state,
    120,
    verbose=True
)

if is_valid:
    print("✅ 代码验证通过")
    # Step 6: 运行回测
    # ... (使用FINSABER回测引擎)

    # 收集结果
    training_results = [
        {
            'sharpe_ratio': 1.234,
            'total_return': 0.456,
            'max_drawdown': 0.123,
            'win_rate': 0.567,
            'code': generated_code,
            'features': ['momentum_5d', 'momentum_10d', 'rsi', 'atr'],
            'feature_importance': {
                'momentum_5d': 0.45,
                'momentum_10d': 0.38,
                'rsi': 0.32,
                'atr': 0.25
            }
        },
        # ... 更多结果
    ]

    # Step 7: 生成COT反馈
    cot_prompt = generator.generate_cot_prompt(
        config,
        training_results,
        iteration=1
    )

    # Step 8: 迭代优化
    # 使用COT提示词调用LLM生成改进代码
    # ...
```

## 四、总结

### 4.1 关键差异总结

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              FINSABER vs LESR 提示词生成关键差异                              │
└─────────────────────────────────────────────────────────────────────────────┘

维度              │ LESR                     │ FINSABER
─────────────────┼──────────────────────────┼────────────────────────────────
任务配置          │ TaskType (Motion/Nav)     │ FinancialTaskType (Timing/Selection)
                  │ 状态维度                  │ 原始状态维度 + 价格历史天数
                  │ 关键维度                  │ 可用数据类型 (OHLCV)
                  │ 性能指标                  │ 性能指标 (Sharpe/Return/Drawdown)
                  │                          │ 风险容忍度 + 市场焦点
─────────────────┼──────────────────────────┼────────────────────────────────
数据描述          │ 物理量 + 单位             │ 金融术语 + 计算公式
                  │ 简洁直观                  │ 详细复杂
                  │ (velocity in m/s)        │ (return, volatility, momentum)
─────────────────┼──────────────────────────┼────────────────────────────────
特征建议          │ 物理特征                 │ 分类技术指标
                  │ 速度、能量、协调          │ 趋势、动量、波动率、成交量
                  │ 3-5个特征                │ 5-15个特征
─────────────────┼──────────────────────────┼────────────────────────────────
奖励原则          │ 单一目标                 │ 多目标权衡
                  │ 前向速度 - 能量           │ 趋势 + 风险 + 成交量
─────────────────┼──────────────────────────┼────────────────────────────────
代码验证          │ 维度、数值稳定性         │ + 金融特定检查
                  │                          │ + Look-ahead bias检测
                  │                          │ + 收益率vs原始价格
─────────────────┼──────────────────────────┼────────────────────────────────
反馈分析          │ Lipschitz常数             │ 相关系数 + SHAP值
                  │ 平滑性                   │ 预测力 + 风险归因
                  │ 成功: 摔倒少              │ 成功: 高Sharpe、低Drawdown
                  │ 失败: 震荡多              │ 失败: 过拟合、过度交易
```

### 4.2 使用建议

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINSABER 提示词使用建议                                    │
└─────────────────────────────────────────────────────────────────────────────┘

✅ 推荐做法:
1. 详细的金融语义解释
2. 分类组织特征建议
3. 提供具体计算示例
4. 多原则奖励设计
5. 金融特定的代码验证
6. 基于相关性的反馈

❌ 避免做法:
1. 直接使用LESR的Prompt
2. 忽略金融术语解释
3. 特征建议过于简单
4. 奖励设计单一目标
5. 忽略look-ahead bias
6. 使用Lipschitz分析

🎯 最佳实践:
1. 从简单的特征组合开始（5-8个）
2. 逐步增加复杂性
3. 重视风险管理
4. 监控过拟合
5. 考虑市场环境
6. 持续迭代优化
```

---

**文档版本**: v1.0
**创建日期**: 2026-04-02
**作者**: LESR-FINSABER集成分析
