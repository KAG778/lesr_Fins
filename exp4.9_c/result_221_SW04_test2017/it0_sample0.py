import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate percentage change in closing prices over the last 20 days
    closing_prices = s[::6]  # Extract closing prices
    pct_change = np.zeros(19)  # There will be 19 changes
    for i in range(1, 20):
        if closing_prices[i - 1] != 0:  # Prevent division by zero
            pct_change[i - 1] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
        else:
            pct_change[i - 1] = 0  # Handle edge case

    features.append(np.mean(pct_change))  # Mean percentage change
    
    # Calculate the volatility: standard deviation of closing prices
    volatility = np.std(closing_prices)
    features.append(volatility)

    # Calculate the current price to moving average ratio
    moving_average = np.mean(closing_prices)  # Simple moving average over 20 days
    current_price = closing_prices[-1]
    price_to_ma_ratio = current_price / moving_average if moving_average != 0 else 0  # Prevent division by zero
    features.append(price_to_ma_ratio)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features (assuming features[0] indicates buy)
        reward -= np.clip(30 * features[0], 30, 50)  # penalty for buy signals
        reward += np.clip(5 * features[1], 5, 10)   # mild reward for sell signals
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 15 * features[0]  # Assuming features[0] indicates a buy signal

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 20 * features[0]  # Assuming features[0] indicates favorable conditions for buy
        elif trend_direction < -0.3:
            reward += 20 * features[1]  # Assuming features[1] indicates favorable conditions for sell

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # assuming features[0] indicates oversold conditions
            reward += 15  # reward for buying in oversold conditions
        if features[1] > 0:  # assuming features[1] indicates overbought conditions
            reward += 15  # reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds