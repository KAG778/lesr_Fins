import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4:120:6]      # Extract volumes for the last 20 days
    days = len(closing_prices)

    # Feature 1: Bollinger Bands Width
    window = 20
    if days >= window:
        rolling_mean = np.mean(closing_prices[-window:])
        rolling_std = np.std(closing_prices[-window:])
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        bb_width = (upper_band - lower_band) / rolling_mean  # Normalize width by rolling mean
    else:
        bb_width = 0

    # Feature 2: Average True Range (ATR) - 14 days
    high_prices = s[1::6]  # Assuming this is the high prices in the same order
    low_prices = s[2::6]   # Assuming this is the low prices in the same order
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                         abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR

    # Feature 3: Rate of Change (ROC)
    roc_period = 10
    roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period] if days >= roc_period else 0

    # Feature 4: Volume Weighted Average Price (VWAP)
    vwap = np.sum(volumes[-20:] * closing_prices[-20:]) / np.sum(volumes[-20:]) if len(volumes) >= 20 and np.sum(volumes[-20:]) != 0 else 0

    # Collect features
    features = [bb_width, atr, roc, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical mean and std for relative thresholds
    historical_mean_risk = 0.5  # Placeholder: Replace with actual historical mean risk
    historical_std_risk = 0.2    # Placeholder: Replace with actual historical std of risk
    
    # Define relative thresholds
    high_risk_threshold = historical_mean_risk + 1.5 * historical_std_risk
    low_risk_threshold = historical_mean_risk - 1.5 * historical_std_risk

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # STRONG NEGATIVE reward for BUY-aligned features
        reward += 20   # Mild positive reward for SELL
        return np.clip(reward, -100, 100)  # Early exit due to high risk
    elif risk_level > low_risk_threshold:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 20 * np.sign(trend_direction)  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude

    return np.clip(reward, -100, 100)