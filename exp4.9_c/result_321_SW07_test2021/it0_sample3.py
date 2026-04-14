import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Feature 1: Price Momentum
    # Calculate momentum as the difference between the closing price of the last day and the first day
    momentum = s[6*19] - s[6*0]  # closing price of the most recent day - closing price of the first day
    features.append(momentum)
    
    # Feature 2: Average Volume
    # Calculate the average trading volume over the last 20 days
    avg_volume = np.mean(s[4::6])  # every 6th entry starting from index 4
    features.append(avg_volume)
    
    # Feature 3: Price Range
    # Calculate the price range as the difference between the maximum high and minimum low over the last 20 days
    price_range = np.max(s[2::6]) - np.min(s[3::6])  # max high - min low
    features.append(price_range)
    
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
        # Strong negative reward for BUY-aligned features
        reward += -40.0  # STRONG NEGATIVE reward for BUY
        # MILD POSITIVE reward for SELL-aligned features
        reward += 7.0  # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20.0  # MODERATE NEGATIVE reward for BUY

    # Check if risk is low before evaluating trend and volatility
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:
                reward += 15.0  # Positive reward for BUY signals in uptrend
            else:
                reward += 15.0  # Positive reward for SELL signals in downtrend
        
        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Assuming features here indicate oversold/overbought conditions
            # E.g. if `features` include an indicator for overbought or oversold
            # reward based on that condition
            # For simplicity, rewarding mean reversion:
            reward += 10.0  # Reward for mean reversion (oversold→buy, overbought→sell)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the bounds