import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4::6]  # Extract trading volumes (every 6th element starting from index 4)
    
    # Feature 1: Price Momentum (Rate of Change)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    
    # Feature 2: Volatility (Standard Deviation of Returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  
    volatility = np.std(returns) if len(returns) > 0 else 0
    
    # Feature 3: Average Volume Change
    if len(volumes) >= 10:
        avg_volume_last_5 = np.mean(volumes[-5:])
        avg_volume_previous_5 = np.mean(volumes[-10:-5])
        volume_change = (avg_volume_last_5 - avg_volume_previous_5) / avg_volume_previous_5 if avg_volume_previous_5 != 0 else 0
    else:
        volume_change = 0
    
    # Feature 4: Relative Strength Index (RSI) Calculation (14-day period)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    average_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    average_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = average_gain / average_loss if average_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if average_loss > 0 else 100

    features = [price_momentum, volatility, volume_change, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate relative thresholds for rewards
    risk_threshold_high = 0.7
    risk_threshold_moderate = 0.4
    trend_threshold = 0.3
    volatility_threshold = 0.6
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY signals
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(5, 15)  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive for upward features
        else:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive for downward features
    
    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > volatility_threshold and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    # Ensure reward stays within bounds of [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward