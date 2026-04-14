import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    N = len(closing_prices)

    # Feature 1: Exponential Moving Average (EMA) over the last 10 days
    ema_period = 10
    if N >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])  # Simplified for demonstration
    else:
        ema = np.nan

    # Feature 2: Average True Range (ATR) for volatility
    def calculate_atr(prices, period=14):
        if len(prices) < period:
            return np.nan
        high = prices[1::6]
        low = prices[2::6]
        close = prices[0::6]
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.nan
        return atr

    atr = calculate_atr(s)  # Use the entire state for ATR calculation

    # Feature 3: Drawdown from the highest price in the last 20 days
    if N >= 20:
        max_price = np.max(closing_prices[-20:])
        current_price = closing_prices[-1]
        drawdown = (max_price - current_price) / max_price
    else:
        drawdown = np.nan

    features = [ema, atr, drawdown]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for BUY-aligned features
        if enhanced_s[123] < 0:  # Assuming negative feature indicates a BUY
            reward += 10  # Mild positive reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20  # Positive reward for upward trend
        else:
            reward += 20  # Positive reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > np.std(enhanced_s[123:126]) and risk_level < 0.4:  # Relative threshold based on std deviation
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return max(-100, min(100, reward))