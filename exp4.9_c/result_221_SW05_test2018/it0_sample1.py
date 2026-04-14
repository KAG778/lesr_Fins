import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Initialize an empty list to hold the new features
    features = []
    
    # Compute the daily returns from closing prices
    closing_prices = s[0::6]  # Extract closing prices
    returns = np.zeros(len(closing_prices) - 1)
    for i in range(1, len(closing_prices)):
        if closing_prices[i-1] != 0:  # Avoid division by zero
            returns[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Feature 1: Average Daily Return
    avg_daily_return = np.mean(returns) if len(returns) > 0 else 0
    features.append(avg_daily_return)

    # Feature 2: Volatility (standard deviation of daily returns)
    volatility = np.std(returns) if len(returns) > 0 else 0
    features.append(volatility)

    # Feature 3: Current Price Relative to Moving Average (20-day moving average)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else np.mean(closing_prices)
    current_price = closing_prices[-1]
    price_relative_to_ma = (current_price - moving_average) / moving_average if moving_average != 0 else 0
    features.append(price_relative_to_ma)

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
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Strong penalty for buying
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)

    # If risk is low, evaluate trend and volatility
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if trend_direction > 0.3:
            # Positive reward for BUY signals
            reward += 10  # Example reward for being in an uptrend
        elif trend_direction < -0.3:
            # Positive reward for SELL signals
            reward += 10  # Example reward for being in a downtrend

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        if abs(trend_direction) < 0.3:
            features = enhanced_state[123:]  # Extract computed features
            # Assume a feature indicating overbought/oversold conditions
            if features[0] < -0.1:  # oversold condition
                reward += 10  # Reward for buying
            elif features[0] > 0.1:  # overbought condition
                reward += 10  # Reward for selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)  # Ensure we return a float value