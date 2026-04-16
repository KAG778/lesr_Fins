"""
Feature Library for Exp4.15

Pure Python + NumPy implementation of 20+ financial indicators with:
- INDICATOR_REGISTRY: name -> {fn, output_dim, default_params, param_ranges, theme}
- build_revise_state(): closure-based assembler from JSON selections to callable
- NormalizedIndicator: Z-score normalization wrapper
- _dedup_by_base_indicator(): same-type dedup keeping highest IC

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
"""

import numpy as np
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Import _extract_json from prompts.py for JSON validation
sys.path.insert(0, str(Path(__file__).parent))
from prompts import _extract_json

# Import ic from metrics.py for screening/stability
from metrics import ic


# ---------------------------------------------------------------------------
# Helper: State extraction
# ---------------------------------------------------------------------------

def _extract_ohlcv(s: np.ndarray):
    """Extract OHLCV arrays from 120d interleaved state.

    Returns: (closes, opens, highs, lows, volumes) as float arrays.
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
# ---------------------------------------------------------------------------

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average using numpy convolution.

    Uses exponential decay weights for proper EMA approximation.
    """
    if len(data) < period or period < 1:
        return np.full_like(data, data[-1] if len(data) > 0 else 0.0)
    alpha = 2.0 / (period + 1.0)
    weights = np.array([(1 - alpha) ** i for i in range(period)])[::-1]
    weights = weights / weights.sum()
    convolved = np.convolve(data, weights, mode='full')[:len(data)]
    return convolved


def _sma(data: np.ndarray, period: int) -> float:
    """Simple moving average of the last `period` values."""
    if len(data) < period or period < 1:
        return float(np.mean(data)) if len(data) > 0 else 0.0
    return float(np.mean(data[-period:]))


# ---------------------------------------------------------------------------
# TREND theme indicators
# ---------------------------------------------------------------------------

def compute_rsi(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Wilder's RSI normalized to [0, 1].

    Returns shape (1,). Neutral default = 0.5 on insufficient data.
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
# ---------------------------------------------------------------------------

def compute_bollinger(s: np.ndarray, window: int = 20, num_std: float = 2.0) -> np.ndarray:
    """Bollinger Band: upper, middle, lower (normalized by price).

    Returns shape (3,).
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
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    returns = np.diff(closes[-(window + 1):])
    vol = np.std(returns) if len(returns) > 0 else 0.0
    return np.array([float(vol)])


# ---------------------------------------------------------------------------
# MEAN_REVERSION theme indicators
# ---------------------------------------------------------------------------

def compute_stochastic(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Stochastic Oscillator %K and %D.

    %K = (close - low_N) / (high_N - low_N) * 100
    %D = SMA(%K, 3) (approximated)

    Returns shape (2,). Values normalized to [0, 1].
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
# ---------------------------------------------------------------------------

def compute_obv(s: np.ndarray) -> np.ndarray:
    """On-Balance Volume normalized by total volume.

    Returns shape (1,).
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
    """
    closes, _, _, _, volumes = _extract_ohlcv(s)
    if len(volumes) < window:
        return np.array([1.0])
    avg_vol = np.mean(volumes[-window:]) + 1e-10
    return np.array([float(volumes[-1] / avg_vol)])


def compute_adx(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Average Directional Index (simplified).

    Returns shape (1,). Value in [0, 1] where 1 = strong trend.
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
# ---------------------------------------------------------------------------

def compute_roc(s: np.ndarray, window: int = 10) -> np.ndarray:
    """Rate of Change: (close[-1] - close[-window]) / close[-window].

    Returns shape (1,).
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
# INDICATOR REGISTRY (D-18: parameterized with ranges)
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
# ---------------------------------------------------------------------------

class NormalizedIndicator:
    """Wraps an indicator function with Z-score normalization.

    Usage:
        ni = NormalizedIndicator(compute_rsi, {'window': 14}, mean=0.5, std=0.1)
        result = ni(raw_state)  # applies (raw - mean) / (std + 1e-8)
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


# ---------------------------------------------------------------------------
# Deduplication (D-08)
# ---------------------------------------------------------------------------

def _dedup_by_base_indicator(selection: List[Dict],
                             ic_scores: Optional[Dict] = None) -> List[Dict]:
    """Remove duplicate base indicators, keeping the one with highest IC.

    Per D-08: if RSI(14) and RSI(21) are both selected, keep only one.

    Args:
        selection: [{"indicator": "RSI", "params": {"window": 14}}, ...]
        ic_scores: {"RSI_14": 0.05, "RSI_21": 0.03, ...} -- keyed by
                   indicator name + sorted params. If None, keeps first.

    Returns:
        Deduplicated selection list.
    """
    groups: Dict[str, List[Dict]] = {}
    for item in selection:
        name = item.get('indicator', '')
        if name not in groups:
            groups[name] = []
        groups[name].append(item)

    result = []
    for name, items in groups.items():
        if len(items) == 1:
            result.append(items[0])
            continue

        if ic_scores is not None:
            # Score each item by its IC
            best_item = items[0]
            best_ic = -float('inf')
            for item in items:
                # Build a key from indicator name + sorted params
                param_parts = [f"{k}_{v}" for k, v in sorted(item.get('params', {}).items())]
                key = f"{name}_{'_'.join(param_parts)}" if param_parts else name
                score = ic_scores.get(key, -float('inf'))
                if score > best_ic:
                    best_ic = score
                    best_item = item
            result.append(best_item)
        else:
            # No IC scores: keep first occurrence
            result.append(items[0])

    return result


# ---------------------------------------------------------------------------
# Validation Pipeline (D-05)
# ---------------------------------------------------------------------------

def validate_selection(json_str: str, sample_state: np.ndarray) -> dict:
    """Multi-stage validation of LLM JSON feature selection.

    Per D-05: parse JSON, check registry, validate params, test on sample data.
    Per D-09: NaN/Inf guard on closure output.

    Args:
        json_str: Raw JSON string from LLM (may contain markdown wrapping).
        sample_state: A sample 120d interleaved state to test the closure on.

    Returns:
        dict with keys:
          - selection: list of valid indicator dicts (with clipped params)
          - revise_state: callable closure (or None if all invalid)
          - feature_dim: int, total output dimensions of valid indicators
          - state_dim: int, 123 + feature_dim
          - errors: list of error/warning strings
    """
    errors = []
    valid_selection = []

    # Stage 1: Parse JSON
    try:
        parsed = _extract_json(json_str)
    except (ValueError, Exception) as e:
        return {
            'selection': [],
            'revise_state': None,
            'feature_dim': 0,
            'state_dim': 123,
            'errors': [f"JSON parse error: {e}"],
        }

    # Stage 2: Check 'features' key
    if 'features' not in parsed:
        return {
            'selection': [],
            'revise_state': None,
            'feature_dim': 0,
            'state_dim': 123,
            'errors': ["Missing 'features' key in JSON"],
        }

    features_list = parsed['features']
    if not isinstance(features_list, list):
        return {
            'selection': [],
            'revise_state': None,
            'feature_dim': 0,
            'state_dim': 123,
            'errors': ["'features' must be a list"],
        }

    if len(features_list) == 0:
        return {
            'selection': [],
            'revise_state': None,
            'feature_dim': 0,
            'state_dim': 123,
            'errors': ["Empty features list -- at least one indicator required"],
        }

    # Stage 3: Validate each feature entry
    for feat in features_list:
        if not isinstance(feat, dict):
            errors.append(f"Invalid feature entry (not a dict): {feat}")
            continue

        name = feat.get('indicator', '')
        params = dict(feat.get('params', {}))

        if name not in INDICATOR_REGISTRY:
            errors.append(f"Unknown indicator: {name}")
            continue

        entry = INDICATOR_REGISTRY[name]

        # Merge defaults with user params
        merged = dict(entry['default_params'])
        merged.update(params)

        # Clip params to registered ranges
        clipped = False
        for pk, pv in merged.items():
            if pk in entry['param_ranges']:
                lo, hi = entry['param_ranges'][pk]
                clipped_val = type(pv)(np.clip(pv, lo, hi))
                if clipped_val != pv:
                    clipped = True
                merged[pk] = clipped_val

        if clipped:
            errors.append(f"Param out of range clipped for {name}: {merged}")

        valid_selection.append({'indicator': name, 'params': merged})

    # Stage 4: If no valid selections remain
    if not valid_selection:
        return {
            'selection': [],
            'revise_state': None,
            'feature_dim': 0,
            'state_dim': 123,
            'errors': errors,
        }

    # Stage 5: Build closure
    revise_fn = build_revise_state(valid_selection)

    # Stage 6: Test on sample_state -- check for NaN/Inf
    try:
        test_output = revise_fn(sample_state)
        if not isinstance(test_output, np.ndarray) or test_output.ndim != 1:
            errors.append("Closure output is not a 1D ndarray")
            return {
                'selection': valid_selection,
                'revise_state': revise_fn,
                'feature_dim': 0,
                'state_dim': 123,
                'errors': errors,
            }
        if np.any(np.isnan(test_output)) or np.any(np.isinf(test_output)):
            errors.append("Closure output contains NaN or Inf values")
            return {
                'selection': valid_selection,
                'revise_state': revise_fn,
                'feature_dim': 0,
                'state_dim': 123,
                'errors': errors,
            }
    except Exception as e:
        errors.append(f"Closure execution error: {e}")
        return {
            'selection': valid_selection,
            'revise_state': revise_fn,
            'feature_dim': 0,
            'state_dim': 123,
            'errors': errors,
        }

    # Compute dimensions
    feature_dim = sum(
        INDICATOR_REGISTRY[s['indicator']]['output_dim']
        for s in valid_selection
    )

    return {
        'selection': valid_selection,
        'revise_state': revise_fn,
        'feature_dim': feature_dim,
        'state_dim': 123 + feature_dim,
        'errors': errors,
    }


# ---------------------------------------------------------------------------
# Feature Screening (D-06, D-07, D-08)
# ---------------------------------------------------------------------------

def screen_features(selection: list, revise_fn, training_states: np.ndarray,
                    forward_returns: np.ndarray) -> dict:
    """Screen features by IC/variance thresholds, dedup, and rank.

    Per D-06: Keep top 5-10 features.
    Per D-07: IC > 0.02, variance > 1e-6.
    Per D-08: Same-type dedup keeps higher IC.

    Args:
        selection: List of {"indicator": ..., "params": ...} dicts.
        revise_fn: Callable from build_revise_state().
        training_states: Array of training states (N x 120).
        forward_returns: Array of forward returns (N,).

    Returns:
        dict with keys:
          - screened_selection: list of feature dicts (5-10, ranked by IC desc)
          - feature_metrics: dict of indicator_name -> {ic, variance}
          - rejected: list of {indicator, params, reason}
    """
    IC_THRESHOLD = 0.02
    VARIANCE_THRESHOLD = 1e-6
    MIN_FEATURES = 5
    MAX_FEATURES = 10

    feature_metrics = {}
    rejected = []

    # Compute features for each indicator separately
    for item in selection:
        name = item.get('indicator', '')
        params = item.get('params', {})

        if name not in INDICATOR_REGISTRY:
            rejected.append({
                'indicator': name,
                'params': params,
                'reason': f"Unknown indicator: {name}",
            })
            continue

        # Build a single-indicator closure to isolate this feature
        single_fn = build_revise_state([item])

        # Compute features across all training states
        try:
            feature_cols = []
            for state in training_states:
                feat = single_fn(state)
                feature_cols.append(feat)
            feature_matrix = np.array(feature_cols)  # (N, output_dim)
        except Exception as e:
            rejected.append({
                'indicator': name,
                'params': params,
                'reason': f"Computation error: {e}",
            })
            continue

        # For multi-output indicators, use the first column for IC
        # but check all columns for variance
        n_outputs = feature_matrix.shape[1] if feature_matrix.ndim > 1 else 1
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(-1, 1)

        # Use mean absolute IC across all output columns
        ics = []
        for col_idx in range(n_outputs):
            col = feature_matrix[:, col_idx]
            col_ic = float(ic(col, forward_returns))
            ics.append(col_ic)
        mean_ic = np.mean(np.abs(ics))
        # Use sign of the strongest IC
        best_ic = ics[np.argmax(np.abs(ics))]

        # Check variance of all output columns
        min_var = float(np.min([
            np.var(feature_matrix[:, j]) for j in range(n_outputs)
        ]))

        feature_metrics[name] = {
            'ic': float(best_ic),
            'variance': float(min_var),
        }

        # Check thresholds
        if abs(best_ic) < IC_THRESHOLD:
            rejected.append({
                'indicator': name,
                'params': params,
                'reason': f"IC={best_ic:.4f} below threshold {IC_THRESHOLD}",
            })
        elif min_var < VARIANCE_THRESHOLD:
            rejected.append({
                'indicator': name,
                'params': params,
                'reason': f"Variance={min_var:.8f} below threshold {VARIANCE_THRESHOLD}",
            })

    # Filter out rejected indicators from the candidates
    rejected_names = {r['indicator'] for r in rejected}
    candidates = [
        item for item in selection
        if item.get('indicator', '') not in rejected_names
    ]

    # Dedup same-type indicators (D-08): keep higher IC
    # Build IC scores dict for dedup
    ic_scores = {}
    for item in candidates:
        name = item.get('indicator', '')
        param_parts = [f"{k}_{v}" for k, v in sorted(item.get('params', {}).items())]
        key = f"{name}_{'_'.join(param_parts)}" if param_parts else name
        ic_scores[key] = feature_metrics.get(name, {}).get('ic', 0.0)

    deduped = _dedup_by_base_indicator(candidates, ic_scores)

    # Sort by IC descending (raw IC value -- higher is better, positive best)
    deduped.sort(
        key=lambda x: feature_metrics.get(x.get('indicator', ''), {}).get('ic', 0.0),
        reverse=True,
    )

    # Keep top 5-10
    screened = deduped[:MAX_FEATURES]

    # Rebuild ranked_metrics to match screened order
    final_metrics = {}
    for item in screened:
        name = item.get('indicator', '')
        if name in feature_metrics:
            final_metrics[name] = feature_metrics[name]

    return {
        'screened_selection': screened,
        'feature_metrics': final_metrics,
        'rejected': rejected,
    }


# ---------------------------------------------------------------------------
# Stability Assessment (D-14, D-15, D-16)
# ---------------------------------------------------------------------------

def assess_stability(selection: list, revise_fn, training_states: np.ndarray,
                     forward_returns: np.ndarray, n_periods: int = 4) -> dict:
    """Sub-period IC stability assessment.

    Per D-14: Split data into n_periods chunks.
    Per D-15: Feature stable if abs(ic_mean) > 0.02 AND ic_std < 2 * abs(ic_mean).
    Per D-16: Report unstable features with reason strings.

    Args:
        selection: List of {"indicator": ..., "params": ...} dicts.
        revise_fn: Callable from build_revise_state().
        training_states: Array of training states (N x 120).
        forward_returns: Array of forward returns (N,).
        n_periods: Number of sub-periods to split into (default 4).

    Returns:
        dict with keys:
          - stability_report: {indicator_name: {ic_per_period, ic_mean, ic_std,
                             stability_score, is_stable}}
          - stable_features: list of stable indicator dicts
          - unstable_features: list of {indicator, params, reason}
    """
    n = len(training_states)
    period_size = n // n_periods

    if period_size < 5:
        # Not enough data for meaningful sub-period analysis
        return {
            'stability_report': {},
            'stable_features': [],
            'unstable_features': [
                {'indicator': s.get('indicator', ''), 'params': s.get('params', {}),
                 'reason': 'Insufficient data for stability analysis'}
                for s in selection
            ],
        }

    stability_report = {}
    stable_features = []
    unstable_features = []

    for item in selection:
        name = item.get('indicator', '')
        params = item.get('params', {})

        if name not in INDICATOR_REGISTRY:
            continue

        # Build single-indicator closure
        single_fn = build_revise_state([item])

        # Compute IC per sub-period
        ic_per_period = []
        for p in range(n_periods):
            start = p * period_size
            end = (p + 1) * period_size if p < n_periods - 1 else n
            sub_states = training_states[start:end]
            sub_returns = forward_returns[start:end]

            # Compute features for this sub-period
            feature_cols = []
            for state in sub_states:
                feat = single_fn(state)
                feature_cols.append(feat)
            feature_matrix = np.array(feature_cols)

            if feature_matrix.ndim == 1:
                feature_matrix = feature_matrix.reshape(-1, 1)

            # Use first output column for IC
            col = feature_matrix[:, 0]
            period_ic = float(ic(col, sub_returns))
            ic_per_period.append(period_ic)

        ic_arr = np.array(ic_per_period)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        stability_score = float(ic_std / (abs(ic_mean) + 1e-8))

        # Stability criterion per D-15
        is_stable = abs(ic_mean) > 0.02 and ic_std < 2 * abs(ic_mean)

        stability_report[name] = {
            'ic_per_period': ic_per_period,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'stability_score': stability_score,
            'is_stable': is_stable,
        }

        if is_stable:
            stable_features.append(item)
        else:
            # Build reason string (per D-16)
            if abs(ic_mean) <= 0.02:
                reason = f"IC mean {ic_mean:.4f} too low (threshold 0.02)"
            else:
                ic_min = float(np.min(ic_arr))
                ic_max = float(np.max(ic_arr))
                reason = f"IC varies from {ic_min:.4f} to {ic_max:.4f} across periods"
            unstable_features.append({
                'indicator': name,
                'params': params,
                'reason': reason,
            })

    return {
        'stability_report': stability_report,
        'stable_features': stable_features,
        'unstable_features': unstable_features,
    }
