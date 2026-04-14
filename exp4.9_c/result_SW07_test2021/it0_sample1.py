import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Price Change Percentage
    price_change_pct = np.zeros(19)  # We will calculate for days 0 to 18
    for i in range(1, 20):
        if closing_prices[i - 1] != 0:  # Avoid division by zero
            price_change_pct[i - 1] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
        else:
            price_change_pct[i - 1] = 0

    # Feature 2: Moving Average of Closing Prices (last 5 days)
    moving_average = np.zeros(20)
    for i in range(4, 20):
        moving_average[i] = np.mean(closing_prices[i - 4:i + 1])

    # Feature 3: Volume Change Percentage
    volume_change_pct = np.zeros(19)  # We will calculate for days 0 to 18
    for i in range(1, 20):
        if volumes[i - 1] != 0:  # Avoid division by zero
            volume_change_pct[i - 1] = (volumes[i] - volumes[i - 1]) / volumes[i - 1]
        else:
            volume_change_pct[i - 1] = 0

    # Combine features into a single array
    features = np.concatenate((price_change_pct, moving_average, volume_change_pct), axis=0)
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
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY signals
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for bullish features
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming features related to mean-reversion are part of enhanced_state[123:]
        mean_reversion_features = enhanced_state[123:]  # Get features
        reward += np.sum(mean_reversion_features[mean_reversion_features > 0])  # Reward oversold
        reward -= np.sum(mean_reversion_features[mean_reversion_features < 0])  # Penalize overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is between -100 and 100