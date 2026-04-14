import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    trading_volume = s[4::6]   # Extract trading volumes

    # Feature 1: Price Change (% from previous closing price)
    price_changes = np.zeros(len(closing_prices))
    for i in range(1, len(closing_prices)):
        if closing_prices[i-1] != 0:
            price_changes[i] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Feature 2: Volume Change (% from previous trading volume)
    volume_changes = np.zeros(len(trading_volume))
    for i in range(1, len(trading_volume)):
        if trading_volume[i-1] != 0:
            volume_changes[i] = (trading_volume[i] - trading_volume[i-1]) / trading_volume[i-1]
    
    # Feature 3: Simple Moving Average of the last 5 closing prices
    moving_avg = np.zeros(len(closing_prices))
    for i in range(4, len(closing_prices)):
        moving_avg[i] = np.mean(closing_prices[i-4:i+1])  # Average of last 5 days

    # Combine features into a single array and return
    features = np.concatenate((price_changes, volume_changes, moving_avg))
    return features[5:]  # Return only from day 5 onwards for moving average

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward
        # Mild positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10)    # MILD POSITIVE reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 10.0  # Moderate negative reward

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += np.random.uniform(10, 20)  # Positive reward for uptrend
        else:
            reward += np.random.uniform(10, 20)  # Positive reward for downtrend

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10.0  # Reward mean-reversion features
        reward -= 5.0   # Penalize breakout-chasing features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]