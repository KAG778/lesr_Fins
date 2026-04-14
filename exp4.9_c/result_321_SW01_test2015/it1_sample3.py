import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    N = len(closing_prices)
    
    # Feature 1: Exponential Moving Average (EMA) over the last 10 days
    ema_period = 10
    if N >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])  # Simplified EMA for illustration
    else:
        ema = np.nan
    
    # Feature 2: Bollinger Bands (Upper, Middle, Lower)
    window = 20
    if N >= window:
        rolling_mean = np.mean(closing_prices[-window:])
        rolling_std = np.std(closing_prices[-window:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
    else:
        upper_band, lower_band = np.nan, np.nan
    
    # Feature 3: Average True Range (ATR) for volatility measurement
    atr_period = 14
    if N >= atr_period:
        high_low = np.max(s[2::6][-atr_period:]) - np.min(s[2::6][-atr_period:])  # High-Low range
        close_prev = s[0::6][-atr_period-1]  # Previous close
        true_range = np.max([high_low, abs(close_prev - s[2::6][-1]), abs(close_prev - s[4::6][-1])])
        atr = np.mean([true_range] + [np.std(s[2::6][-atr_period:])])  # Simplified ATR calculation
    else:
        atr = np.nan

    features = [ema, upper_band, lower_band, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        if enhanced_s[123] >= 0:  # Assuming positive feature indicates a BUY
            return reward  # Early return as risk is high
        else:
            reward += np.random.uniform(5, 10)  # MILD POSITIVE for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive reward for alignment with upward momentum
        elif trend_direction < 0:  # Downtrend
            reward += 20  # Positive reward for alignment with downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion during sideways market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > np.std(enhanced_s[123:]) and risk_level < 0.4:  # Using relative threshold
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return max(-100, min(100, reward))