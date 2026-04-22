"""
Market Regime Detector for Exp4.9

Actual raw state structure (120d):
  s[i*6 + 0] = close  (i = 0..19)
  s[i*6 + 1] = open
  s[i*6 + 2] = high
  s[i*6 + 3] = low
  s[i*6 + 4] = volume
  s[i*6 + 5] = adj_close

Regime dimensions:
  [0] trend_strength:     [-1, +1]
  [1] volatility_regime:  [0, 1]
  [2] momentum_signal:    [-1, +1]
  [3] meanrev_signal:     [-1, +1]
  [4] crisis_signal:      [0, 1]
"""

import numpy as np
from typing import Tuple


def _parse_state(s: np.ndarray):
    """Parse 120d interleaved state into per-field arrays."""
    closes = np.array([s[i*6] for i in range(20)], dtype=float)
    opens = np.array([s[i*6 + 1] for i in range(20)], dtype=float)
    highs = np.array([s[i*6 + 2] for i in range(20)], dtype=float)
    lows = np.array([s[i*6 + 3] for i in range(20)], dtype=float)
    volumes = np.array([s[i*6 + 4] for i in range(20)], dtype=float)
    adj_closes = np.array([s[i*6 + 5] for i in range(20)], dtype=float)
    return closes, opens, highs, lows, volumes, adj_closes


def detect_regime(s: np.ndarray) -> np.ndarray:
    """
    Compute 5-dimensional regime vector from 120d raw state.
    """
    closes, opens, highs, lows, volumes, adj_closes = _parse_state(s)

    trend_strength = _compute_trend_strength(closes)
    volatility_regime = _compute_volatility_regime(closes, highs, lows)
    momentum_signal = _compute_momentum_signal(closes)
    meanrev_signal = _compute_meanrev_signal(closes)
    crisis_signal = _compute_crisis_signal(closes, highs, lows, volumes)

    return np.array([
        trend_strength,
        volatility_regime,
        momentum_signal,
        meanrev_signal,
        crisis_signal
    ], dtype=float)


def classify_regime(regime_vector: np.ndarray) -> str:
    """Classify dominant regime."""
    trend, vol, momentum, meanrev, crisis = regime_vector
    if crisis > 0.5:
        return 'crisis'
    if vol > 0.8:
        return 'high_vol'
    if trend > 0.3:
        return 'trend_up'
    if trend < -0.3:
        return 'trend_down'
    return 'sideways'


def _compute_trend_strength(closes: np.ndarray) -> float:
    if len(closes) < 5 or np.std(closes) < 1e-8:
        return 0.0
    ma5 = np.mean(closes[-5:])
    ma20 = np.mean(closes)
    normalized = (ma5 - ma20) / (ma20 + 1e-8)
    # Consistency
    recent = closes[-5:]
    up_days = np.sum(np.diff(recent) > 0)
    consistency = (up_days - 2) / 2.0
    strength = 0.6 * np.clip(normalized * 20, -1, 1) + 0.4 * consistency
    return float(np.clip(strength, -1, 1))


def _compute_volatility_regime(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
    n = len(closes)
    if n < 5:
        return 0.5

    # Simple: use recent daily range relative to price
    daily_ranges = (highs - lows) / (closes + 1e-8)
    current_range = np.mean(daily_ranges[-5:])
    hist_range = np.mean(daily_ranges) if len(daily_ranges) > 5 else current_range
    hist_std = np.std(daily_ranges) if len(daily_ranges) > 5 else 0

    if hist_std < 1e-10:
        return 0.5

    z_score = (current_range - hist_range) / hist_std
    return float(np.clip((z_score + 1) / 3, 0, 1))


def _compute_momentum_signal(closes: np.ndarray) -> float:
    n = len(closes)
    if n < 10:
        return 0.0

    # Multi-period ROC
    rocs = []
    for period in [3, 5, 10]:
        if n > period:
            roc = (closes[-1] - closes[-1 - period]) / (closes[-1 - period] + 1e-8) * 100
            rocs.append(roc)
    if not rocs:
        return 0.0

    avg_roc = np.mean(rocs)

    # Z-score against historical
    hist_rocs = []
    for i in range(5, n):
        r = (closes[i] - closes[i-5]) / (closes[i-5] + 1e-8) * 100
        hist_rocs.append(r)
    if len(hist_rocs) < 5:
        return float(np.clip(avg_roc / 5, -1, 1))

    mean_r = np.mean(hist_rocs)
    std_r = np.std(hist_rocs)
    if std_r < 1e-10:
        return 0.0
    z = (avg_roc - mean_r) / std_r
    return float(np.clip(z / 2, -1, 1))


def _compute_meanrev_signal(closes: np.ndarray) -> float:
    n = len(closes)
    if n < 10:
        return 0.0

    ma = np.mean(closes[-20:]) if n >= 20 else np.mean(closes)
    std = np.std(closes[-20:]) if n >= 20 else np.std(closes)
    if std < 1e-10:
        return 0.0

    upper = ma + 2 * std
    lower = ma - 2 * std
    pct_b = (closes[-1] - lower) / (upper - lower + 1e-8)
    signal = (pct_b - 0.5) * 2
    return float(np.clip(signal, -1, 1))


def _compute_crisis_signal(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> float:
    n = len(closes)
    if n < 5:
        return 0.0

    # Factor 1: Recent max drawdown (within 10 days)
    window = min(10, n)
    recent_high = np.max(closes[-window:])
    current = closes[-1]
    recent_dd = (recent_high - current) / (recent_high + 1e-8)

    # Factor 2: Volume spike
    if n >= 10 and np.mean(volumes[-10:-1]) > 0:
        vol_ratio = volumes[-1] / (np.mean(volumes[-10:-1]) + 1e-8)
    else:
        vol_ratio = 1.0

    # Factor 3: Consecutive down days
    consecutive_down = 0
    for i in range(len(closes) - 1, max(0, len(closes) - 6), -1):
        if closes[i] < closes[i - 1]:
            consecutive_down += 1
        else:
            break

    dd_score = np.clip(recent_dd / 0.15, 0, 1)  # 15% dd → 1.0 (raised threshold from 10%)
    vol_score = np.clip((vol_ratio - 2.0) / 3.0, 0, 1)  # 5x volume → 1.0
    consec_score = np.clip(consecutive_down / 5.0, 0, 1)

    crisis = max(dd_score, 0.5 * vol_score + 0.5 * consec_score)
    return float(np.clip(crisis, 0, 1))
