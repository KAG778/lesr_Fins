import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract volumes
    
    # Feature 1: Exponential Moving Average (EMA) Divergence
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else np.mean(closing_prices)
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else np.mean(closing_prices)
    ema_divergence = short_ema - long_ema

    # Feature 2: Average True Range (ATR) for volatility
    if len(closing_prices) < 2:
        atr = 0
    else:
        high_prices = s[2::6]  # High prices
        low_prices = s[3::6]   # Low prices
        tr_values = np.maximum(high_prices[-1] - low_prices[-1], 
                                np.abs(high_prices[-1] - closing_prices[-2]), 
                                np.abs(low_prices[-1] - closing_prices[-2]))
        atr = np.mean(tr_values[-14:]) if len(tr_values) >= 14 else np.mean(tr_values)

    # Feature 3: Bollinger Bands Width
    rolling_mean = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else np.mean(closing_prices)
    rolling_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bollinger_width = rolling_std * 2  # 2 standard deviations for Bollinger Bands

    # Feature 4: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0

    features = [ema_divergence, atr, bollinger_width, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Determine the historical std for relative thresholds
    historical_std = np.std(enhanced_s[0:120]) if np.std(enhanced_s[0:120]) != 0 else 1
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 100 * (risk_level - 0.7)  # Strong negative for BUY-aligned features
        reward += 20 * (1 - risk_level)  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 50 * (risk_level - 0.4)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 50 * (trend_direction)  # Positive reward for upward features
        elif trend_direction < 0:  # Downtrend
            reward += 50 * (-trend_direction)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 30  # Reward for mean-reversion features
        reward -= 10 * (1 - risk_level)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward