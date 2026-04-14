import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4::6]          # Extract trading volumes (every 6th element starting from index 4)

    # Feature 1: Average True Range (ATR) - a measure of volatility
    high_prices = s[1::6]      # High prices at index 1, 7, 13, ...
    low_prices = s[2::6]       # Low prices at index 2, 8, 14, ...
    
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 2: Price Position in Bollinger Bands
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices[-20:]) > 0 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices[-20:]) > 0 else 0
    upper_band = moving_average + (2 * std_dev)
    lower_band = moving_average - (2 * std_dev)
    price_position = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0  # Value between 0 and 1

    # Feature 3: Change in Volume Relative to Moving Average
    volume_moving_average = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
    volume_change = (volumes[-1] - volume_moving_average) / volume_moving_average if volume_moving_average != 0 else 0

    # Feature 4: Momentum of Closing Prices (Rate of Change)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Combine features into a single array
    features = [atr, price_position, volume_change, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate thresholds based on historical data
    risk_threshold_high = 0.7
    risk_threshold_moderate = 0.4
    trend_threshold = 0.3
    volatility_threshold = 0.6

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(10, 20)   # Mild positive for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(5, 15)    # Mild negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        reward += np.random.uniform(10, 20) if trend_direction > 0 else -np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward