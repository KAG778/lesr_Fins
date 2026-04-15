"""
Market Regime Detector for Exp4.9_c

3-dimensional regime vector (simpler than exp4.9's 5-dim):
  [0] trend_direction:  [-1, +1]  -1=downtrend, 0=sideways, +1=uptrend
  [1] volatility_level: [0, 1]    0=calm, 1=extreme
  [2] risk_level:       [0, 1]    0=safe, 1=high risk (recent large drop)

State layout (120d interleaved):
  s[i*6 + 0] = close,  s[i*6 + 1] = open,
  s[i*6 + 2] = high,   s[i*6 + 3] = low,
  s[i*6 + 4] = volume, s[i*6 + 5] = adj_close
  for i = 0..19 (20 trading days)
"""

import numpy as np


def detect_regime(s: np.ndarray) -> np.ndarray:
    """Compute 3-dimensional regime vector from 120d raw state."""
    n = len(s) // 6
    closes = np.array([s[i*6] for i in range(n)], dtype=float)
    highs = np.array([s[i*6+2] for i in range(n)], dtype=float)
    lows = np.array([s[i*6+3] for i in range(n)], dtype=float)
    volumes = np.array([s[i*6+4] for i in range(n)], dtype=float)

    trend = _trend_direction(closes)
    volatility = _volatility_level(closes, highs, lows)
    risk = _risk_level(closes)

    return np.array([trend, volatility, risk], dtype=float)


def _trend_direction(closes: np.ndarray) -> float:
    """MA(5) vs MA(20) relative distance, clipped to [-1, 1]."""
    if len(closes) < 5 or np.std(closes) < 1e-8:
        return 0.0
    ma5 = np.mean(closes[-5:])
    ma20 = np.mean(closes)
    trend = (ma5 - ma20) / (np.mean(closes) * 0.05 + 1e-8)
    return float(np.clip(trend, -1, 1))


def _volatility_level(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
    """Recent daily range relative to historical, z-scored to [0, 1]."""
    n = len(closes)
    if n < 5:
        return 0.5
    ranges = (highs - lows) / (closes + 1e-8)
    recent = np.mean(ranges[-5:])
    hist_mean = np.mean(ranges)
    hist_std = np.std(ranges) + 1e-10
    z = (recent - hist_mean) / hist_std
    return float(np.clip((z + 1) / 3, 0, 1))


def _risk_level(closes: np.ndarray) -> float:
    """Max drawdown in recent 10 days, scaled to [0, 1]."""
    n = len(closes)
    if n < 3:
        return 0.0
    window = closes[-min(10, n):]
    recent_high = np.max(window)
    current = closes[-1]
    dd = (recent_high - current) / (recent_high + 1e-8)
    return float(np.clip(dd / 0.15, 0, 1))  # 15% drop → 1.0
