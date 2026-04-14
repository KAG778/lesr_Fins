import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Average True Range (ATR) - a measure of volatility
    high_prices = s[1::6]      # High prices at index 1, 7, 13, ...
    low_prices = s[2::6]       # Low prices at index 2, 8, 14, ...
    
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 2: Bollinger Bands
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices[-20:]) > 0 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices[-20:]) > 0 else 0
    upper_band = moving_average + (2 * std_dev)
    lower_band = moving_average - (2 * std_dev)
    price_position = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0  # Value between 0 and 1
    
    # Feature 3: Change in Volume Relative to Moving Average
    volume_moving_average = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
    volume_change = (volumes[-1] - volume_moving_average) / volume_moving_average if volume_moving_average != 0 else 0
    
    features = [atr, price_position, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Define thresholds based on historical standard deviation
    risk_thresholds = (0.2, 0.5, 0.7)  # Example thresholds for risk levels
    trend_threshold = 0.3  # Example threshold for trend direction
    
    reward = 0.0
    
    # Priority 1: Risk Management
    if risk_level > risk_thresholds[2]:  # Risk level high
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY-aligned features
        reward += np.random.uniform(10, 20)   # Mild positive for SELL signals
    elif risk_level > risk_thresholds[1]:  # Moderate risk
        reward -= np.random.uniform(5, 15)    # Moderate negative for BUY signals
    
    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < risk_thresholds[1]:  # Trend is strong and risk is low
        reward += np.random.uniform(10, 20) if trend_direction > 0 else -np.random.uniform(10, 20)  # Positive reward for momentum alignment
    
    # Priority 3: Sideways Market / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < risk_thresholds[0]:  # Sideways and low risk
        reward += np.random.uniform(5, 15)  # Reward for mean reversion
    
    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_thresholds[1]:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    # Ensure reward stays within bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward