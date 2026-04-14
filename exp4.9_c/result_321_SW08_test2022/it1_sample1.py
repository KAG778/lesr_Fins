import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Reshape the raw state into a 20-day view with 6 features each
    days = s.reshape((20, 6))

    # Feature 1: Average Daily Return
    closing_prices = days[:, 0]  # Closing prices
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices  # Daily return as a percentage
    avg_daily_return = np.mean(daily_returns)  # Average daily return
    features.append(avg_daily_return)

    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    # Feature 3: Relative Strength Index (RSI) (14-day)
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 4: Price Momentum (Current Close - Previous Close)
    momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0
    features.append(momentum)

    # Feature 5: Drawdown from the highest price in the last 20 days
    max_price = np.max(closing_prices)
    drawdown = (max_price - closing_prices[-1]) / max_price if max_price > 0 else 0
    features.append(drawdown)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0  # Initialize reward

    # Calculate relative thresholds based on historical data
    avg_volatility = np.mean(features[1])  # Using the second feature (volatility) for relative threshold
    std_volatility = np.std(features[1])

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -50  # Strong penalty for high risk BUY
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Moderate penalty for elevated risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[3] > 0:  # Uptrend and positive momentum
            reward += 20  # Strong positive reward for correct direction
        elif trend_direction < -0.3 and features[3] < 0:  # Downtrend and negative momentum
            reward += 20  # Strong positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3:
        if risk_level < 0.3:
            reward += 10  # Reward mean-reversion features when risk is low
        else:
            reward -= 5  # Penalize chasing trends in sideways market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > (avg_volatility + 1.5 * std_volatility) and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward