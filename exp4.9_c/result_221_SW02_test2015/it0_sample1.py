import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Extract closing prices
    volumes = s[4::6]        # Extract trading volumes
    num_days = len(closing_prices)

    # Feature 1: Price Change Percentage
    price_changes = np.zeros(num_days - 1)
    for i in range(1, num_days):
        if closing_prices[i-1] != 0:  # Avoid division by zero
            price_changes[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Feature 2: Simple Moving Average (last 5 days)
    moving_average = np.zeros(num_days)
    for i in range(4, num_days):
        moving_average[i] = np.mean(closing_prices[i-4:i+1])  # Average of last 5 days

    # Feature 3: Volume Change Percentage
    volume_changes = np.zeros(num_days - 1)
    for i in range(1, num_days):
        if volumes[i-1] != 0:  # Avoid division by zero
            volume_changes[i-1] = (volumes[i] - volumes[i-1]) / volumes[i-1]

    # Combine features into a single array
    features = np.concatenate((price_changes, moving_average, volume_changes[1:]))  # volume_changes has one less element
    return features

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
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY
        reward += np.random.uniform(5, 10)   # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:
        reward -= 10  # moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        features = enhanced_state[123:]  # Your computed features
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Assuming features[0] indicates upward price momentum
                reward += 10  # Positive reward for correct trend-following action
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Assuming features[0] indicates downward price momentum
                reward += 10  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        features = enhanced_state[123:]  # Your computed features
        if features[0] < 0:  # Assuming features[0] indicates oversold condition
            reward += 10  # Reward for buying in oversold condition
        elif features[0] > 0:  # Assuming features[0] indicates overbought condition
            reward -= 10  # Penalize for breakout-chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward