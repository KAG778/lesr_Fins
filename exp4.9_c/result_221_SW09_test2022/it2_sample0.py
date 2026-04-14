import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: 14-Day Exponential Moving Average (EMA) of closing prices
    ema = np.mean(closing_prices[-14:]) if len(closing_prices) >= 14 else 0
    
    # Feature 2: Price Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) >= 15 and closing_prices[-15] != 0 else 0
    
    # Feature 3: Z-Score of the last 20 closing prices
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:])
        z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    else:
        z_score = 0
    
    # Feature 4: Volume Weighted Average Price (VWAP)
    if len(volumes) >= 10:
        vwap = np.sum(closing_prices[-10:] * volumes[-10:]) / np.sum(volumes[-10:]) if np.sum(volumes[-10:]) != 0 else 0
    else:
        vwap = 0

    features = [ema, roc, z_score, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Compute relative thresholds based on historical data
    risk_threshold_high = 0.7  # Example high risk threshold
    risk_threshold_moderate = 0.4  # Example moderate risk threshold
    trend_threshold = 0.3  # Example trend direction threshold

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY signals
        reward += np.random.uniform(10, 20)   # Mild positive for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(5, 15)    # Mild negative for BUY signals

    # Priority 2 — TREND FOLLOWING (only if risk is low)
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        reward += np.random.uniform(10, 20) if trend_direction > 0 else -np.random.uniform(10, 20)  # Positive for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward