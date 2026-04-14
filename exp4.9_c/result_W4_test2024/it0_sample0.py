import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices (index 0, 6, 12, ...)
    volumes = s[4::6]  # Trading volumes (index 4, 10, 16, ...)
    
    # Feature 1: Daily Percentage Change in Closing Price
    price_change = np.diff(closing_prices) / closing_prices[:-1]  # Avoid division by zero
    
    # Handle cases where the percentage change might have NaN values
    price_change = np.nan_to_num(price_change)  # Replace NaN with 0
    
    # Feature 2: Average Volume
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    
    # Feature 3: Bollinger Bands Width (using a rolling window)
    rolling_mean = np.mean(closing_prices)  # Simple mean over the entire period
    rolling_std = np.std(closing_prices)  # Standard deviation over the entire period
    bollinger_upper = rolling_mean + (2 * rolling_std)
    bollinger_lower = rolling_mean - (2 * rolling_std)
    bollinger_width = bollinger_upper - bollinger_lower
    
    features = [
        np.mean(price_change),  # Average daily price change
        avg_volume,             # Average trading volume
        bollinger_width         # Width of Bollinger Bands
    ]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += np.random.uniform(-50, -30)  # STRONG NEGATIVE for BUY
        reward += np.random.uniform(5, 10)     # MILD POSITIVE for SELL
    elif risk_level > 0.4:
        reward += np.random.uniform(-20, -5)   # Moderate negative for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for favorable trend
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming we have some features indicating overbought/oversold
        # Using features[123:] for additional logic can be implemented here
        reward += 5  # Reward for mean-reversion logic

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clamp the reward to be within the specified range
    return np.clip(reward, -100, 100)