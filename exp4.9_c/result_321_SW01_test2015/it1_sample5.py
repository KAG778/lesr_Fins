import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    N = len(closing_prices)

    # Feature 1: Exponential Moving Average (EMA) over the last 5 days
    ema_period = 5
    if N >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])  # Using mean as a simple approximation for EMA
    else:
        ema = np.nan
    
    # Feature 2: Rate of Change (ROC)
    roc_period = 5
    if N > roc_period:
        roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period]  # Rate of Change
    else:
        roc = np.nan
    
    # Feature 3: Average True Range (ATR) for volatility
    if N > 1:
        high_low = np.diff(s[1::6])  # High - Low
        high_close = np.abs(np.diff(s[1::6]))  # High - Previous Close
        low_close = np.abs(np.diff(s[2::6]))  # Low - Previous Close
        true_ranges = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.nan
    else:
        atr = np.nan
    
    # Feature 4: Volume Weighted Average Price (VWAP)
    if N > 0:
        vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else np.nan
    else:
        vwap = np.nan

    # Combine features into a single array
    features = [ema, roc, atr, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Normalize thresholds based on historical std deviation
    risk_threshold_high = 0.7
    risk_threshold_mid = 0.4
    trend_threshold = 0.3
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for high risk when BUY
        return max(-100, min(100, reward))  # Early return
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_mid:
        if trend_direction > 0:
            reward += 30  # Strong positive reward for bullish trend
        else:
            reward += 30  # Strong positive reward for bearish trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return max(-100, min(100, reward))