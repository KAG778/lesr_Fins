import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extracting closing prices
    trading_volumes = s[4:120:6]  # Extracting trading volumes
    
    # Feature 1: Price Change Ratio (current price - previous price) / previous price
    price_change_ratio = np.zeros(20)  # Initialize array for price change ratios
    for i in range(1, len(closing_prices)):
        if closing_prices[i-1] != 0:  # Prevent division by zero
            price_change_ratio[i] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Feature 2: Average Trading Volume over the last 20 days
    average_volume = np.mean(trading_volumes)

    # Feature 3: Relative Strength Index (RSI) calculation
    deltas = np.diff(closing_prices)  # Calculate price changes
    gain = np.where(deltas > 0, deltas, 0)  # Gains
    loss = -np.where(deltas < 0, deltas, 0)  # Losses
    
    avg_gain = np.mean(gain[-14:])  # Average gain over the last 14 days
    avg_loss = np.mean(loss[-14:])  # Average loss over the last 14 days
    
    # Prevent division by zero in RSI calculation
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    
    # Compile features
    features = [price_change_ratio[-1], average_volume, rsi]  # Return only the last day's features
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40  # Midpoint of the range (-30 to -50)
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Adjust as needed for the range

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Positive reward for aligning with the trend
        if trend_direction > 0:
            reward += 20  # Positive reward for BUY aligned with upward trend
        else:
            reward += 20  # Positive reward for SELL aligned with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += 15  # Reward for mean-reversion signals
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)

# Note: The reward function might need to be adjusted based on the specific context and features being used.