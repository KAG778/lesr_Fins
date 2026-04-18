"""
Market-Level State Detector for Portfolio Optimization

Computes 6-dimensional market state vector from equal-weight portfolio of all 5 stocks.
  [0] trend_direction:    [-1, +1]   5-day MA vs overall mean
  [1] volatility_level:   [0, 1]     recent vol z-score vs historical
  [2] risk_level:         [0, 1]     max drawdown depth in recent 10 days
  [3] avg_cross_corr:     [-1, 1]    mean of 5x5 pairwise return correlation
  [4] market_breadth:     [0, 1]     fraction of stocks with positive 5d return
  [5] volatility_ratio:   [0, 3]     recent 5d vol / recent 20d vol

Input: dict of 5 raw states {ticker: 120d_array}
"""

import numpy as np

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']


def _extract_closes(s: np.ndarray) -> np.ndarray:
    n = len(s) // 6
    return np.array([s[i * 6] for i in range(n)], dtype=float)


def detect_market_regime(raw_states: dict) -> np.ndarray:
    """Compute 6-dim market state from equal-weight portfolio.

    Args:
        raw_states: dict {ticker: 120d_array}

    Returns:
        6-dim numpy array [trend, vol_level, risk, avg_corr, breadth, vol_ratio]
    """
    all_closes = []
    for ticker in TICKERS:
        if ticker in raw_states:
            all_closes.append(_extract_closes(raw_states[ticker]))

    if not all_closes:
        return np.array([0.0, 0.5, 0.0, 0.0, 0.5, 1.0])

    min_len = min(len(c) for c in all_closes)
    if min_len < 5:
        return np.array([0.0, 0.5, 0.0, 0.0, 0.5, 1.0])

    aligned = np.array([c[:min_len] for c in all_closes])
    port_closes = np.mean(aligned, axis=0)

    trend = _trend_direction(port_closes)
    volatility = _volatility_level(port_closes)
    risk = _risk_level(port_closes)
    avg_corr = _avg_cross_correlation(aligned)
    breadth = _market_breadth(aligned)
    vol_ratio = _volatility_ratio(aligned)

    return np.array([trend, volatility, risk, avg_corr, breadth, vol_ratio], dtype=float)


def _trend_direction(closes: np.ndarray) -> float:
    if len(closes) < 5 or np.std(closes) < 1e-8:
        return 0.0
    ma5 = np.mean(closes[-5:])
    ma_all = np.mean(closes)
    trend = (ma5 - ma_all) / (np.mean(closes) * 0.05 + 1e-8)
    return float(np.clip(trend, -1, 1))


def _volatility_level(closes: np.ndarray) -> float:
    if len(closes) < 5:
        return 0.5
    returns = np.diff(closes) / (closes[:-1] + 1e-8)
    recent_vol = np.std(returns[-5:])
    hist_std = np.std(returns) * 0.5 + 1e-10
    z = (recent_vol - np.std(returns)) / hist_std
    return float(np.clip((z + 1) / 3, 0, 1))


def _risk_level(closes: np.ndarray) -> float:
    if len(closes) < 3:
        return 0.0
    window = closes[-min(10, len(closes)):]
    recent_high = np.max(window)
    current = closes[-1]
    dd = (recent_high - current) / (recent_high + 1e-8)
    return float(np.clip(dd / 0.15, 0, 1))


def _avg_cross_correlation(aligned_closes: np.ndarray) -> float:
    """Mean of pairwise return correlations across all stocks."""
    n_stocks = aligned_closes.shape[0]
    if n_stocks < 2:
        return 0.0
    all_returns = []
    for i in range(n_stocks):
        c = aligned_closes[i]
        if len(c) < 2:
            return 0.0
        all_returns.append(np.diff(c) / (c[:-1] + 1e-10))
    pair_corrs = []
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            r1, r2 = all_returns[i], all_returns[j]
            n = min(len(r1), len(r2))
            if n < 5:
                continue
            s1, s2 = np.std(r1[-n:]), np.std(r2[-n:])
            if s1 < 1e-10 or s2 < 1e-10:
                continue
            corr = float(np.mean((r1[-n:] - np.mean(r1[-n:])) * (r2[-n:] - np.mean(r2[-n:]))) / (s1 * s2))
            pair_corrs.append(corr)
    if not pair_corrs:
        return 0.0
    return float(np.clip(np.mean(pair_corrs), -1, 1))


def _market_breadth(aligned_closes: np.ndarray) -> float:
    """Fraction of stocks with positive 5-day return."""
    n_stocks = aligned_closes.shape[0]
    positive = 0
    for i in range(n_stocks):
        c = aligned_closes[i]
        if len(c) < 6:
            continue
        ret_5d = (c[-1] - c[-6]) / (c[-6] + 1e-10)
        if ret_5d > 0:
            positive += 1
    return float(positive / n_stocks)


def _volatility_ratio(aligned_closes: np.ndarray) -> float:
    """Ratio of recent 5-day vol to recent 20-day vol for equal-weight portfolio."""
    port_closes = np.mean(aligned_closes, axis=0)
    if len(port_closes) < 21:
        return 1.0
    returns = np.diff(port_closes) / (port_closes[:-1] + 1e-10)
    if len(returns) < 20:
        return 1.0
    vol_5 = np.std(returns[-5:], ddof=1) + 1e-10
    vol_20 = np.std(returns[-20:], ddof=1) + 1e-10
    return float(np.clip(vol_5 / vol_20, 0, 3))
