import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extracting closing prices
    n_days = 5  # For features
    
    # Feature 1: Price Change (percentage change over the last n_days)
    price_change = np.zeros(19)  # because we only have 20 days, the first 4 days won't have change
    for i in range(1, len(closing_prices) - n_days):
        if closing_prices[i - 1] != 0:  # Prevent division by zero
            price_change[i] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
    
    # Feature 2: Moving Average (Simple Moving Average over last n_days)
    moving_average = np.zeros(20)
    for i in range(n_days - 1, len(closing_prices)):
        moving_average[i] = np.mean(closing_prices[i - n_days + 1:i + 1])
    
    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    average_gain = np.zeros(20)
    average_loss = np.zeros(20)
    
    # Calculate average gains and losses
    for i in range(n_days, len(closing_prices)):
        average_gain[i] = np.mean(gain[i - n_days + 1:i + 1])
        average_loss[i] = np.mean(loss[i - n_days + 1:i + 1])
    
    rs = average_gain / (average_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Return new features
    features = np.concatenate((price_change, moving_average, rsi))
    return features[4:]  # Return only valid features (after dropping leading zeros)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strongly discourage buying in dangerous conditions
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        return reward  # Early exit
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(5, 15)
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Consider upward features (e.g., price change)
            reward += np.random.uniform(5, 15)  # Positive reward for BUY
        elif trend_direction < -0.3:
            # Consider downward features
            reward += np.random.uniform(5, 15)  # Positive reward for SELL
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward for mean-reversion features
        reward += np.random.uniform(5, 15)  # Positive reward for HOLD if mean-reversion is detected
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]