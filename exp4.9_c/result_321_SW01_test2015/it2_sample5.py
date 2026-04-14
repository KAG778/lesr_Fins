import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    N = len(closing_prices)

    # Feature 1: 10-Day Exponential Moving Average (EMA)
    ema_period = 10
    if N >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])  # Simple average as an approximation
    else:
        ema = np.nan

    # Feature 2: Average True Range (ATR)
    def calculate_atr(prices, period=14):
        if len(prices) < period:
            return np.nan
        high = prices[1::6]
        low = prices[2::6]
        close = prices[0::6]
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-period:])

    atr = calculate_atr(s)  # Use the entire state for ATR calculation

    # Feature 3: Rate of Change (ROC)
    roc_period = 5
    if N > roc_period:
        roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period]  # Rate of Change
    else:
        roc = np.nan

    # Feature 4: 20-Day Moving Average for trend detection
    ma_20 = np.mean(closing_prices[-20:]) if N >= 20 else np.nan

    # Feature 5: Volume Weighted Average Price (VWAP)
    if N > 0:
        vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else np.nan
    else:
        vwap = np.nan

    features = [ema, atr, roc, ma_20, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    risk_threshold_high = 0.7
    risk_threshold_mid = 0.4
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY-aligned features
        return max(-100, min(100, reward))  # Early return if risk is high

    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_mid:
        reward += 30 * (1 if trend_direction > 0 else -1)  # Strong positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > np.std(enhanced_s[123:126]) and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return max(-100, min(100, reward))