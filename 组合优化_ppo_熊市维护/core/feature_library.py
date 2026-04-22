"""
Feature Library for Portfolio PPO

Pure Python + NumPy implementation of 20+ financial indicators with:
- INDICATOR_REGISTRY: name -> {fn, output_dim, default_params, param_ranges, theme}
- build_revise_state(): closure-based assembler from JSON selections to callable
- NormalizedIndicator: Z-score normalization wrapper

Design decisions (from CONTEXT.md):
- D-19: Pure Python + NumPy only, no ta-lib or pandas_ta
- D-21: Closure-based assembly, no exec/eval
- D-17: Z-score normalization on indicator outputs
- D-18: Parameterized indicators with validated ranges
- D-09: NaN/Inf guards on every indicator and closure

State layout (120d interleaved, from regime_detector.py):
  s[i*6 + 0] = close,  s[i*6 + 1] = open,
  s[i*6 + 2] = high,   s[i*6 + 3] = low,
  s[i*6 + 4] = volume, s[i*6 + 5] = adj_close
  for i = 0..19 (20 trading days)

Note: validate_selection, screen_features, assess_stability, and
_dedup_by_base_indicator have been moved to lesr_controller.py with
portfolio-aware logic.

特征库模块 —— 投资组合 PPO 的技术指标计算

本模块用纯 Python + NumPy 实现了 20+ 个金融技术指标，分为四类：
- TREND（趋势类）：RSI, MACD, EMA交叉, 动量, ROC 等
- VOLATILITY（波动率类）：布林带, ATR, 波动率, 偏度, 峰度
- MEAN_REVERSION（均值回归类）：随机指标, 威廉指标, CCI
- VOLUME（成交量类）：OBV, 量比, ADX

此外还提供 9 个"构建块"函数（Building Blocks），供 LLM 生成的代码直接 import 使用。

状态布局（120维交错格式）：
  s[i*6 + 0] = 收盘价,  s[i*6 + 1] = 开盘价,
  s[i*6 + 2] = 最高价,  s[i*6 + 3] = 最低价,
  s[i*6 + 4] = 成交量,  s[i*6 + 5] = 复权收盘价
  其中 i = 0..19（20个交易日）

关键设计决策：
- D-19: 纯 Python + NumPy，不依赖 ta-lib 或 pandas_ta
- D-21: 基于闭包的状态组装，不使用 exec/eval
- D-17: 对指标输出做 Z-score 归一化
- D-18: 参数化指标，带验证范围
- D-09: 每个指标和闭包都有 NaN/Inf 防护
"""

import numpy as np
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helper: State extraction
# 状态提取辅助函数：从 120 维交错状态中提取各通道数据
# ---------------------------------------------------------------------------

def _extract_ohlcv(s: np.ndarray):
    """Extract OHLCV arrays from 120d interleaved state.

    Returns: (closes, opens, highs, lows, volumes) as float arrays.

    从 120 维交错状态数组中提取 OHLCV 数据。
    输入 s 的布局为 [close, open, high, low, volume, adj_close] * 20天。
    """
    n = len(s) // 6
    closes = np.array([s[i * 6 + 0] for i in range(n)], dtype=float)
    opens = np.array([s[i * 6 + 1] for i in range(n)], dtype=float)
    highs = np.array([s[i * 6 + 2] for i in range(n)], dtype=float)
    lows = np.array([s[i * 6 + 3] for i in range(n)], dtype=float)
    volumes = np.array([s[i * 6 + 4] for i in range(n)], dtype=float)
    return closes, opens, highs, lows, volumes


# ---------------------------------------------------------------------------
# Helper: Moving averages
# 移动平均线辅助函数：EMA 和 SMA
# ---------------------------------------------------------------------------

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average using numpy convolution.

    Uses exponential decay weights for proper EMA approximation.

    指数移动平均线（EMA），使用 numpy 卷积实现。
    利用指数衰减权重近似 EMA。
    """
    if len(data) < period or period < 1:
        return np.full_like(data, data[-1] if len(data) > 0 else 0.0)
    alpha = 2.0 / (period + 1.0)
    weights = np.array([(1 - alpha) ** i for i in range(period)])[::-1]
    weights = weights / weights.sum()
    convolved = np.convolve(data, weights, mode='full')[:len(data)]
    return convolved


def _sma(data: np.ndarray, period: int) -> float:
    """Simple moving average of the last `period` values.

    简单移动平均线（SMA）：取最后 period 个值的均值。
    """
    if len(data) < period or period < 1:
        return float(np.mean(data)) if len(data) > 0 else 0.0
    return float(np.mean(data[-period:]))


# ---------------------------------------------------------------------------
# TREND theme indicators
# 趋势类指标（5个）：RSI, MACD, EMA交叉, 动量, ROC
# 用于捕捉价格趋势的方向和强度
# ---------------------------------------------------------------------------

def compute_rsi(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Wilder's RSI normalized to [0, 1].

    Returns shape (1,). Neutral default = 0.5 on insufficient data.

    Wilder's RSI（相对强弱指标），归一化到 [0, 1]。
    输出形状 (1,)。数据不足时返回中性默认值 0.5。
    RSI > 0.7 表示超买，< 0.3 表示超卖。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.5])
    deltas = np.diff(closes[-(window + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) + 1e-10
    rs = avg_gain / avg_loss
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return np.array([rsi_val / 100.0])


def compute_macd(s: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD line, Signal line, Histogram.

    Returns shape (3,). Zeros on insufficient data.

    MACD 指标（移动平均收敛散度），输出 3 个维度：MACD线、信号线、柱状图。
    输出按价格归一化，数据不足时返回零。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < slow:
        return np.zeros(3)
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    macd_val = float(macd_line[-1])

    # Signal line: EMA of MACD line (use last `signal` values)
    if len(macd_line) >= signal:
        signal_val = float(_ema(macd_line, signal)[-1])
    else:
        signal_val = 0.0

    histogram = macd_val - signal_val
    # Normalize by recent price to get reasonable scale
    price = closes[-1] if closes[-1] != 0 else 1.0
    return np.array([macd_val / price, signal_val / price, histogram / price])


def compute_ema_cross(s: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    """EMA crossover signal: (EMA_fast - EMA_slow) / price.

    Returns shape (1,). Positive = bullish cross, negative = bearish.

    EMA 交叉信号：(快线 EMA - 慢线 EMA) / 价格。
    正值表示看涨交叉（金叉），负值表示看跌交叉（死叉）。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < slow:
        return np.array([0.0])
    ema_fast_val = float(_ema(closes, fast)[-1])
    ema_slow_val = float(_ema(closes, slow)[-1])
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    cross = (ema_fast_val - ema_slow_val) / price
    return np.array([cross])


def compute_momentum(s: np.ndarray, window: int = 10) -> np.ndarray:
    """Rate of change / momentum: (close[-1] - close[-window]) / close[-window].

    Returns shape (1,).

    动量指标（变化率）：当前收盘价相对 window 天前收盘价的涨跌幅。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    denom = closes[-window - 1]
    if abs(denom) < 1e-10:
        return np.array([0.0])
    mom = (closes[-1] - closes[-window - 1]) / denom
    return np.array([float(mom)])


# ---------------------------------------------------------------------------
# VOLATILITY theme indicators
# 波动率类指标（3个）：布林带, ATR, 波动率
# 用于衡量价格波动程度和风险水平
# ---------------------------------------------------------------------------

def compute_bollinger(s: np.ndarray, window: int = 20, num_std: float = 2.0) -> np.ndarray:
    """Bollinger Band: upper, middle, lower (normalized by price).

    Returns shape (3,).

    布林带指标：上轨、中轨、下轨（按价格归一化）。
    输出 3 个维度，反映价格相对统计波动范围的位置。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.zeros(3)
    recent = closes[-window:]
    sma = np.mean(recent)
    std = np.std(recent) + 1e-10
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    upper = (sma + num_std * std - closes[-1]) / price
    middle = (sma - closes[-1]) / price
    lower = (sma - num_std * std - closes[-1]) / price
    return np.array([upper, middle, lower])


def compute_atr(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Average True Range normalized by price.

    Returns shape (1,).

    平均真实波幅（ATR），按价格归一化。
    衡量价格波动的绝对幅度，常用于设置止损或评估风险。
    """
    closes, _, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    atr_val = np.mean(tr[-window:])
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([float(atr_val / price)])


def compute_volatility(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling standard deviation of returns.

    Returns shape (1,).

    滚动波动率：收益率的滚动标准差。
    直接衡量近期收益率的波动程度。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    returns = np.diff(closes[-(window + 1):])
    vol = np.std(returns) if len(returns) > 0 else 0.0
    return np.array([float(vol)])


# ---------------------------------------------------------------------------
# MEAN_REVERSION theme indicators
# 均值回归类指标（3个）：随机指标, 威廉%R, CCI
# 用于识别价格偏离均值后的回归信号
# ---------------------------------------------------------------------------

def compute_stochastic(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Stochastic Oscillator %K and %D.

    %K = (close - low_N) / (high_N - low_N) * 100
    %D = SMA(%K, 3) (approximated)

    Returns shape (2,). Values normalized to [0, 1].

    随机振荡器（Stochastic Oscillator）%K 和 %D。
    %K = (收盘价 - N日最低价) / (N日最高价 - N日最低价)
    %D = %K 的 3 日简单移动平均
    输出 2 个维度，归一化到 [0, 1]。
    """
    closes, _, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.5, 0.5])
    recent_high = np.max(highs[-window:])
    recent_low = np.min(lows[-window:])
    price_range = recent_high - recent_low
    if abs(price_range) < 1e-10:
        return np.array([0.5, 0.5])
    pct_k = (closes[-1] - recent_low) / price_range
    # Approximate %D as average of recent %K values
    if len(closes) >= window + 2:
        pk_values = []
        for offset in range(min(3, len(closes) - window)):
            h = np.max(highs[-(window + offset):len(highs) - offset if offset > 0 else len(highs)])
            l = np.min(lows[-(window + offset):len(lows) - offset if offset > 0 else len(lows)])
            r = h - l
            if abs(r) > 1e-10:
                pk_values.append((closes[-(1 + offset)] - l) / r)
            else:
                pk_values.append(0.5)
        pct_d = np.mean(pk_values)
    else:
        pct_d = pct_k
    return np.array([float(pct_k), float(pct_d)])


def compute_williams_r(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Williams %R: (high_N - close) / (high_N - low_N) * -100, normalized to [0, 1].

    Returns shape (1,).

    威廉 %R 指标，归一化到 [0, 1]。
    0 表示超买（接近最高价），1 表示超卖（接近最低价）。
    """
    closes, _, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.5])
    recent_high = np.max(highs[-window:])
    recent_low = np.min(lows[-window:])
    price_range = recent_high - recent_low
    if abs(price_range) < 1e-10:
        return np.array([0.5])
    wr = (recent_high - closes[-1]) / price_range
    return np.array([float(wr)])  # [0, 1] where 0=overbought, 1=oversold


def compute_cci(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Commodity Channel Index normalized to [-1, 1] range.

    Returns shape (1,).

    商品通道指标（CCI），归一化到 [-1, 1]。
    正值表示价格高于统计均值，负值表示低于均值，绝对值越大偏离越远。
    """
    closes, opens, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.0])
    typical_prices = (highs + lows + closes) / 3.0
    recent_tp = typical_prices[-window:]
    sma_tp = np.mean(recent_tp)
    mean_dev = np.mean(np.abs(recent_tp - sma_tp)) + 1e-10
    cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_dev)
    # Normalize to roughly [-1, 1] range (CCI can be very large)
    return np.array([float(np.clip(cci / 200.0, -1.0, 1.0))])


# ---------------------------------------------------------------------------
# VOLUME theme indicators
# 成交量类指标（3个）：OBV, 量比, ADX
# 用于从成交量角度分析价格趋势的可靠性
# ---------------------------------------------------------------------------

def compute_obv(s: np.ndarray) -> np.ndarray:
    """On-Balance Volume normalized by total volume.

    Returns shape (1,).

    能量潮指标（OBV），按总成交量归一化。
    通过累积上涨日和下跌日的成交量差来衡量买卖压力。
    """
    closes, _, _, _, volumes = _extract_ohlcv(s)
    if len(closes) < 2:
        return np.array([0.0])
    total_vol = np.sum(volumes) + 1e-10
    obv = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
    return np.array([float(obv / total_vol)])


def compute_volume_ratio(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Current volume / average volume over window.

    Returns shape (1,).

    量比：当前成交量 / 过去 window 天的平均成交量。
    值 > 1 表示放量，< 1 表示缩量。
    """
    closes, _, _, _, volumes = _extract_ohlcv(s)
    if len(volumes) < window:
        return np.array([1.0])
    avg_vol = np.mean(volumes[-window:]) + 1e-10
    return np.array([float(volumes[-1] / avg_vol)])


def compute_adx(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Average Directional Index (simplified).

    Returns shape (1,). Value in [0, 1] where 1 = strong trend.

    平均方向指数（ADX），简化版。
    值在 [0, 1] 范围内，越接近 1 表示趋势越强。
    ADX 不区分趋势方向，只衡量趋势的强度。
    """
    closes, _, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])

    # True range
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )

    # Directional movement
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smooth with window
    if len(tr) < window:
        return np.array([0.0])
    atr_val = np.mean(tr[-window:]) + 1e-10
    plus_di = 100.0 * np.mean(plus_dm[-window:]) / atr_val
    minus_di = 100.0 * np.mean(minus_dm[-window:]) / atr_val

    denom = plus_di + minus_di + 1e-10
    dx = 100.0 * abs(plus_di - minus_di) / denom
    # Normalize to [0, 1]
    return np.array([float(np.clip(dx / 100.0, 0.0, 1.0))])


# ---------------------------------------------------------------------------
# EXTENDED indicators (to reach 20+)
# 扩展指标（7个）：ROC, SMA交叉, DEMA, 偏度, 峰度, 威廉鳄鱼, TSF
# 补充基础指标未能覆盖的技术分析维度
# ---------------------------------------------------------------------------

def compute_roc(s: np.ndarray, window: int = 10) -> np.ndarray:
    """Rate of Change: (close[-1] - close[-window]) / close[-window].

    Returns shape (1,).

    变化率指标（ROC）：当前价相对 window 天前的涨跌百分比。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    denom = closes[-window - 1]
    if abs(denom) < 1e-10:
        return np.array([0.0])
    roc = (closes[-1] - denom) / denom
    return np.array([float(roc)])


def compute_sma_cross(s: np.ndarray, fast: int = 10, slow: int = 30) -> np.ndarray:
    """SMA crossover signal: (SMA_fast - SMA_slow) / price.

    Returns shape (1,).

    SMA 交叉信号：(快线 SMA - 慢线 SMA) / 价格。
    正值表示短期均线上穿长期均线（看涨），负值反之。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < slow:
        return np.array([0.0])
    sma_fast = np.mean(closes[-fast:])
    sma_slow = np.mean(closes[-slow:])
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([float((sma_fast - sma_slow) / price)])


def compute_dema(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Double EMA: 2*EMA - EMA(EMA), normalized by price.

    Returns shape (1,).

    双重指数移动平均线（DEMA）：2*EMA - EMA(EMA)，按价格归一化。
    比 EMA 反应更快，减少滞后性。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.0])
    ema1 = _ema(closes, window)
    ema2 = _ema(ema1, window)
    dema_val = 2.0 * ema1[-1] - ema2[-1]
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([float(dema_val / price)])


def compute_skewness(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Return distribution skewness over window.

    Returns shape (1,).

    收益率分布偏度。
    正偏表示右偏（有极端正收益），负偏表示左偏（有极端负收益）。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    returns = np.diff(closes[-(window + 1):])
    if len(returns) < 3:
        return np.array([0.0])
    std = np.std(returns) + 1e-10
    mean = np.mean(returns)
    skew = np.mean(((returns - mean) / std) ** 3)
    return np.array([float(np.clip(skew, -5.0, 5.0))])


def compute_kurtosis(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Return distribution kurtosis over window (excess kurtosis).

    Returns shape (1,).

    收益率分布超额峰度。
    正值表示厚尾分布（极端事件概率更高），负值表示薄尾。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    returns = np.diff(closes[-(window + 1):])
    if len(returns) < 4:
        return np.array([0.0])
    std = np.std(returns) + 1e-10
    mean = np.mean(returns)
    kurt = np.mean(((returns - mean) / std) ** 4) - 3.0  # excess kurtosis
    return np.array([float(np.clip(kurt, -5.0, 10.0))])


def compute_williams_alligator(s: np.ndarray) -> np.ndarray:
    """Williams Alligator: jaw (13), teeth (8), lips (5) SMAs, normalized by price.

    Returns shape (3,).

    威廉鳄鱼指标：颚（13日SMA）、齿（8日SMA）、唇（5日SMA），按价格归一化。
    三线分散表示趋势形成，三线纠缠表示盘整。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < 13:
        return np.zeros(3)
    jaw = float(np.mean(closes[-13:]))
    teeth = float(np.mean(closes[-8:]))
    lips = float(np.mean(closes[-5:]))
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([jaw / price, teeth / price, lips / price])


def compute_tsf(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Time Series Forecast: linear regression slope * window + intercept, normalized.

    Returns shape (1,).

    时间序列预测（TSF）：线性回归斜率 * window + 截距，按价格归一化。
    反映价格的线性趋势方向和强度。
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.0])
    recent = closes[-window:]
    x = np.arange(window, dtype=float)
    slope = np.polyfit(x, recent, 1)[0]
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([float(slope * window / price)])


# ---------------------------------------------------------------------------
# Building-Block Functions for LLM Code Import
# 构建块函数 —— 供 LLM 生成的代码直接 import 使用
# ---------------------------------------------------------------------------
# LLM 生成的 revise_state 代码可以通过以下方式导入这些函数：
#   from feature_library import compute_relative_momentum, compute_realized_volatility, ...
#
# 设计思路：提供 9 个独立的、输入输出明确的基础函数，让 LLM 无需从零实现
# 金融指标计算，只需选择合适的函数组合即可。这降低了 LLM 代码出错的概率。

def compute_relative_momentum(prices: np.ndarray, window: int = 20) -> float:
    """Excess return of this stock vs the window-average return.

    相对动量：个股在 window 天内的收益率。
    用于识别相对表现优异的股票。
    """
    if len(prices) < window + 1 or prices[-window - 1] == 0:
        return 0.0
    return float((prices[-1] - prices[-window - 1]) / abs(prices[-window - 1]))


def compute_cross_sectional_rank(values: list) -> float:
    """Rank of a single value among all stocks' values. Returns [0, 1].

    Note: Portfolio-level only, not inside revise_state(s).

    截面排名：某个值在所有股票值中的排名，返回 [0, 1]。
    注意：这是组合级函数，不能在 revise_state(s) 内使用（因为 revise_state 只处理单只股票）。
    """
    if not values or len(values) < 2:
        return 0.5
    target = values[0]
    rank = sum(1 for v in values if v <= target)
    return float(rank / len(values))


def compute_realized_volatility(returns: np.ndarray, window: int = 20) -> float:
    """Realized volatility (std of returns) over window.

    已实现波动率：window 天内收益率的标准差。
    衡量个股风险水平。
    """
    if len(returns) < window:
        window = len(returns)
    if window < 2:
        return 0.0
    return float(np.std(returns[-window:]))


def compute_downside_risk(returns: np.ndarray, window: int = 20) -> float:
    """Downside semi-deviation over window.

    下行风险（下行半偏差）：只考虑负收益的标准差。
    相比普通波动率，更关注损失侧的风险。
    """
    if len(returns) < window:
        window = len(returns)
    if window < 2:
        return 0.0
    neg = returns[-window:]
    neg = neg[neg < 0]
    if len(neg) < 2:
        return 0.0
    return float(np.std(neg))


def compute_beta(returns: np.ndarray, market_returns: np.ndarray,
                 window: int = 20) -> float:
    """Rolling beta to market (equal-weight portfolio).

    滚动 Beta 系数：个股收益对等权组合收益的敏感度。
    Beta > 1 表示个股波动大于市场，< 1 表示波动小于市场。
    """
    n = min(len(returns), len(market_returns), window)
    if n < 5:
        return 1.0
    r = returns[-n:]
    m = market_returns[-n:]
    var_m = np.var(m)
    if var_m < 1e-10:
        return 1.0
    cov = np.mean((r - np.mean(r)) * (m - np.mean(m)))
    return float(cov / var_m)


def compute_multi_horizon_momentum(prices: np.ndarray,
                                   windows: list = None) -> np.ndarray:
    """Momentum at multiple time horizons.

    多时间尺度动量：在多个窗口（默认 5/10/20 天）分别计算动量。
    输出 3 个维度，捕捉短期、中期、长期趋势。
    """
    if windows is None:
        windows = [5, 10, 20]
    result = []
    for w in windows:
        if len(prices) > w and prices[-w - 1] != 0:
            result.append((prices[-1] - prices[-w - 1]) / abs(prices[-w - 1]))
        else:
            result.append(0.0)
    return np.array(result, dtype=float)


def compute_zscore_price(prices: np.ndarray, window: int = 20) -> float:
    """Z-score of current price vs N-day mean.

    价格 Z-score：当前价格相对 N 日均值的标准化偏离程度。
    正值表示价格高于均值，负值表示低于均值。用作均值回归信号。
    """
    if len(prices) < window:
        return 0.0
    seg = prices[-window:]
    mean_val = np.mean(seg)
    std_val = np.std(seg) + 1e-10
    return float(np.clip((prices[-1] - mean_val) / std_val, -3, 3))


def compute_mean_reversion_signal(prices: np.ndarray, window: int = 20) -> float:
    """Mean reversion strength: how far price deviated and started returning.

    均值回归强度：衡量价格偏离均值后开始回归的程度。
    通过前一日 Z-score 与当前 Z-score 的差值来捕捉回归信号。
    """
    if len(prices) < window + 2:
        return 0.0
    seg = prices[-window:]
    mean_val = np.mean(seg)
    std_val = np.std(seg) + 1e-10
    z_current = (prices[-1] - mean_val) / std_val
    z_prev = (prices[-2] - mean_val) / std_val
    return float(np.clip(z_prev - z_current, -3, 3))


def compute_turnover_ratio(volumes: np.ndarray, window: int = 20) -> float:
    """Current volume / average volume ratio.

    换手率比：当前成交量 / 过去 window 天的平均成交量。
    用于检测放量/缩量，衡量流动性。
    """
    if len(volumes) < window + 1:
        return 1.0
    avg = np.mean(volumes[-window - 1:-1]) + 1e-10
    return float(volumes[-1] / avg)


BUILDING_BLOCKS = [
    ('compute_relative_momentum', 'prices, window=20', 1,
     "Excess return vs window-average. Identifies outperforming stocks."),
    ('compute_cross_sectional_rank', 'values', 1,
     "Rank among all stocks [0,1]. Portfolio-level only, not in revise_state."),
    ('compute_realized_volatility', 'returns, window=20', 1,
     "Realized volatility. Measure individual stock risk."),
    ('compute_downside_risk', 'returns, window=20', 1,
     "Downside semi-deviation. Measure downside risk."),
    ('compute_beta', 'returns, market_returns, window=20', 1,
     "Beta to equal-weight portfolio. Systemic risk exposure."),
    ('compute_multi_horizon_momentum', 'prices, windows=[5,10,20]', 3,
     "Multi-period momentum. Capture trends at multiple scales."),
    ('compute_zscore_price', 'prices, window=20', 1,
     "Price z-score vs N-day mean. Mean reversion signal."),
    ('compute_mean_reversion_signal', 'prices, window=20', 1,
     "Mean reversion strength. Identify overextended prices."),
    ('compute_turnover_ratio', 'volumes, window=20', 1,
     "Volume ratio. Liquidity detection."),
]


# ---------------------------------------------------------------------------
# INDICATOR REGISTRY (D-18: parameterized with ranges)
# 指标注册表：每个指标记录函数、输出维度、默认参数、参数范围和主题
# 用于参数验证和自动组装
# ---------------------------------------------------------------------------

INDICATOR_REGISTRY: Dict[str, dict] = {
    # --- TREND theme (5) ---
    'RSI': {
        'fn': compute_rsi,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 60)},
        'theme': 'trend',
    },
    'MACD': {
        'fn': compute_macd,
        'output_dim': 3,
        'default_params': {'fast': 12, 'slow': 26, 'signal': 9},
        'param_ranges': {'fast': (5, 20), 'slow': (15, 60), 'signal': (3, 15)},
        'theme': 'trend',
    },
    'EMA_Cross': {
        'fn': compute_ema_cross,
        'output_dim': 1,
        'default_params': {'fast': 12, 'slow': 26},
        'param_ranges': {'fast': (5, 20), 'slow': (15, 60)},
        'theme': 'trend',
    },
    'Momentum': {
        'fn': compute_momentum,
        'output_dim': 1,
        'default_params': {'window': 10},
        'param_ranges': {'window': (5, 60)},
        'theme': 'trend',
    },
    'ROC': {
        'fn': compute_roc,
        'output_dim': 1,
        'default_params': {'window': 10},
        'param_ranges': {'window': (5, 60)},
        'theme': 'trend',
    },
    # --- VOLATILITY theme (3) ---
    'Bollinger': {
        'fn': compute_bollinger,
        'output_dim': 3,
        'default_params': {'window': 20, 'num_std': 2.0},
        'param_ranges': {'window': (10, 40), 'num_std': (1.0, 3.0)},
        'theme': 'volatility',
    },
    'ATR': {
        'fn': compute_atr,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'volatility',
    },
    'Volatility': {
        'fn': compute_volatility,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 60)},
        'theme': 'volatility',
    },
    # --- MEAN_REVERSION theme (3) ---
    'Stochastic': {
        'fn': compute_stochastic,
        'output_dim': 2,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'mean_reversion',
    },
    'Williams_R': {
        'fn': compute_williams_r,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'mean_reversion',
    },
    'CCI': {
        'fn': compute_cci,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 30)},
        'theme': 'mean_reversion',
    },
    # --- VOLUME theme (3) ---
    'OBV': {
        'fn': compute_obv,
        'output_dim': 1,
        'default_params': {},
        'param_ranges': {},
        'theme': 'volume',
    },
    'Volume_Ratio': {
        'fn': compute_volume_ratio,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 30)},
        'theme': 'volume',
    },
    'ADX': {
        'fn': compute_adx,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'volume',
    },
    # --- EXTENDED indicators (7 more = 21 total) ---
    'SMA_Cross': {
        'fn': compute_sma_cross,
        'output_dim': 1,
        'default_params': {'fast': 10, 'slow': 30},
        'param_ranges': {'fast': (5, 20), 'slow': (15, 60)},
        'theme': 'trend',
    },
    'DEMA': {
        'fn': compute_dema,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 60)},
        'theme': 'trend',
    },
    'Skewness': {
        'fn': compute_skewness,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 60)},
        'theme': 'volatility',
    },
    'Kurtosis': {
        'fn': compute_kurtosis,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 60)},
        'theme': 'volatility',
    },
    'Williams_Alligator': {
        'fn': compute_williams_alligator,
        'output_dim': 3,
        'default_params': {},
        'param_ranges': {},
        'theme': 'trend',
    },
    'TSF': {
        'fn': compute_tsf,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'trend',
    },
}


# ---------------------------------------------------------------------------
# Z-score Normalization Wrapper (D-17)
# Z-score 归一化包装器：对指标输出进行标准化
# ---------------------------------------------------------------------------

class NormalizedIndicator:
    """Wraps an indicator function with Z-score normalization.

    Usage:
        ni = NormalizedIndicator(compute_rsi, {'window': 14}, mean=0.5, std=0.1)
        result = ni(raw_state)  # applies (raw - mean) / (std + 1e-8)

    Z-score 归一化指标包装器。
    对指标输出进行标准化：(原始值 - 均值) / (标准差 + 1e-8)。
    使用方法：
        ni = NormalizedIndicator(compute_rsi, {'window': 14}, mean=0.5, std=0.1)
        result = ni(raw_state)
    """

    def __init__(self, fn: Callable, params: dict,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        self.fn = fn
        self.params = params
        self.mean = mean
        self.std = std

    def __call__(self, raw_state: np.ndarray) -> np.ndarray:
        raw = self.fn(raw_state, **self.params)
        if self.mean is not None and self.std is not None:
            return (raw - self.mean) / (self.std + 1e-8)
        return raw


# ---------------------------------------------------------------------------
# Closure-based assembler (D-21: no exec/eval)
# 基于闭包的状态组装器：将选定的指标组合成一个可调用函数
# ---------------------------------------------------------------------------

def build_revise_state(selection: List[Dict]) -> Callable:
    """Build a closure that computes all selected features.

    Per D-21: closure-based assembly, no exec/eval.
    Per D-18: params clipped to registered param_ranges.
    Per D-09: NaN/Inf replaced with zeros.

    Args:
        selection: [{"indicator": "RSI", "params": {"window": 14}}, ...]

    Returns:
        Callable that takes raw_state (120d) and returns 1D feature array.

    基于闭包组装 revise_state 函数。
    设计原则：
    - D-21: 使用闭包而非 exec/eval，更安全
    - D-18: 参数裁剪到注册范围内，防止极端参数
    - D-09: NaN/Inf 自动替换为零，保证数值稳定

    参数：
        selection: 指标选择列表，如 [{"indicator": "RSI", "params": {"window": 14}}, ...]

    返回：
        可调用函数，接受 raw_state (120维)，返回 1维特征数组
    """
    funcs = []
    output_dims = []

    for item in selection:
        name = item.get('indicator', '')
        params = dict(item.get('params', {}))  # copy to avoid mutation

        if name not in INDICATOR_REGISTRY:
            continue  # silently skip unknown indicators

        entry = INDICATOR_REGISTRY[name]

        # Merge defaults with user-specified params (user overrides defaults)
        merged = dict(entry['default_params'])
        merged.update(params)

        # Clip params to registered ranges (D-18, T-03-01)
        for pk, pv in merged.items():
            if pk in entry['param_ranges']:
                lo, hi = entry['param_ranges'][pk]
                merged[pk] = type(pv)(np.clip(pv, lo, hi))  # preserve type

        funcs.append((entry['fn'], merged))
        output_dims.append(entry['output_dim'])

    if not funcs:
        # Fallback: return zeros(3) when no valid indicators
        def revise_state_fallback(raw_state: np.ndarray) -> np.ndarray:
            return np.zeros(3)
        return revise_state_fallback

    # Capture funcs and output_dims in closure
    _funcs = funcs
    _output_dims = output_dims

    def revise_state(raw_state: np.ndarray) -> np.ndarray:
        features = []
        for idx, (fn, params) in enumerate(_funcs):
            try:
                result = fn(raw_state, **params)
                # Ensure result is 1D numpy array
                if not isinstance(result, np.ndarray):
                    result = np.atleast_1d(np.array(result, dtype=float))
                if result.ndim != 1:
                    result = result.flatten()
                # NaN/Inf guard (D-09, T-03-02)
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    result = np.zeros(_output_dims[idx])
                features.append(result)
            except Exception:
                # Graceful fallback: zeros of correct dimension
                features.append(np.zeros(_output_dims[idx]))

        if not features:
            return np.zeros(3)
        return np.concatenate(features)

    return revise_state
