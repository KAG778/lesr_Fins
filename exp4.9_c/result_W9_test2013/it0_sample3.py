import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    closing_prices = s[0::6]  # extract closing prices
    opening_prices = s[1::6]  # extract opening prices
    high_prices = s[2::6]     # extract high prices
    low_prices = s[3::6]      # extract low prices
    volumes = s[4::6]         # extract volumes
    
    # Calculate features
    features = []
    
    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_change_pct)
    
    # Feature 2: Moving Average (10-day)
    moving_average = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    features.append(moving_average)
    
    # Feature 3: Volume Change (current vs previous day)
    volume_change_pct = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
    features.append(volume_change_pct)
    
    # Feature 4: Price Volatility (standard deviation of last 5 days)
    price_volatility = np.std(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    features.append(price_volatility)

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strongly negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        # Mild positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for uptrend
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (not implemented in features, but if we assume some)
        reward += 5  # Reward for mean-reversion behavior

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(reward, 100))

    return reward