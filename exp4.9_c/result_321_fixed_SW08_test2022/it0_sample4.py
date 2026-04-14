import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extracting the closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    volumes = s[4:120:6]  # Trading volumes for 20 days

    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume Over Last 20 Days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: Closing Price vs. 5-Day Moving Average
    moving_average_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    closing_vs_moving_avg = closing_prices[-1] - moving_average_5

    # Return the features as a numpy array
    features = [price_change_pct, average_volume, closing_vs_moving_avg]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 45.0  # Strong negative reward for buying in dangerous conditions
    elif risk_level > 0.4:
        reward -= 15.0  # Moderate negative reward for buying in elevated risk conditions

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        # If in an uptrend, reward positive features
        if trend_direction > 0:
            reward += features[0] * 50.0  # Price Change Percentage positively aligned
        # If in a downtrend, reward negative features
        else:
            reward += -features[0] * 50.0  # Price Change Percentage negatively aligned

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward for mean-reversion features, penalize for breakout chasing
        if features[2] > 0:  # Closing price is above moving average
            reward += -5.0  # Penalizing breakout chasing
        else:  # Closing price is below moving average
            reward += 10.0  # Reward for mean-reversion

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))