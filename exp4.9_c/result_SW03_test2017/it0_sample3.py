import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract closing prices
    closing_prices = s[0:120:6]  # Every 6th element starting from index 0 is the closing price

    # Feature 1: Price Change Percentage from 5 days ago
    if closing_prices[15] != 0:  # Avoid division by zero
        price_change_percentage = (closing_prices[19] - closing_prices[15]) / closing_prices[15]
    else:
        price_change_percentage = 0

    # Feature 2: Historical Volatility (20-day)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns) * np.sqrt(20)  # Annualized volatility

    # Feature 3: Moving Average Divergence (5-day vs 20-day)
    short_moving_average = np.mean(closing_prices[-5:])  # Last 5 days
    long_moving_average = np.mean(closing_prices[-20:])  # Last 20 days
    moving_average_divergence = short_moving_average - long_moving_average

    # Return the computed features
    features = [price_change_percentage, historical_volatility, moving_average_divergence]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += -np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # Positive reward for upward features
        else:
            reward += 10  # Positive reward for downward features (correct bearish bet)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)    # Reward for mean-reversion features
        reward += -np.random.uniform(5, 15)    # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)