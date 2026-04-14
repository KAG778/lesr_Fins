import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # take every 6th element starting from index 0 for closing prices
    volumes = s[4::6]  # take every 6th element starting from index 4 for volumes

    # Compute features
    features = []

    # Feature 1: Price Momentum (current close - close 5 days ago)
    if len(closing_prices) > 5:
        price_momentum = closing_prices[0] - closing_prices[5]
    else:
        price_momentum = 0  # Handle edge case

    features.append(price_momentum)

    # Feature 2: Volume Change (percentage change from previous day)
    if len(volumes) > 1 and volumes[1] != 0:
        volume_change = (volumes[0] - volumes[1]) / volumes[1]
    else:
        volume_change = 0  # Handle edge case

    features.append(volume_change)

    # Feature 3: Relative Strength Index (RSI)
    if len(closing_prices) > 14:  # RSI typically uses a 14-day period
        delta = np.diff(closing_prices[:14])
        gain = np.sum(delta[delta > 0]) / 14
        loss = -np.sum(delta[delta < 0]) / 14
        rs = gain / loss if loss != 0 else 0  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Default RSI if not enough data

    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_state):
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Random strong negative reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # uptrend
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds