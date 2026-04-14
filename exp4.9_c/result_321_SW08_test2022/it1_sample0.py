import numpy as np

def revise_state(s):
    features = []

    # Reshape the raw state into a 20-day view with 6 features each
    days = s.reshape((20, 6))

    # Feature 1: Daily return volatility (standard deviation of daily returns)
    closing_prices = days[:, 0]  # Closing prices
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices
    vol = np.std(daily_returns)

    # Feature 2: 14-day Average True Range (ATR) for volatility measurement
    high_prices = days[:, 3]
    low_prices = days[:, 4]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0

    # Feature 3: 14-day Exponential Moving Average (EMA) of closing prices
    ema = np.mean(closing_prices[-14:]) if len(closing_prices) >= 14 else 0

    # Feature 4: Change in the number of advancing vs. declining stocks (Breadth)
    # This requires additional data input, assuming it's available as "advancing_declining" array.
    # For the sake of this example, we will simulate it:
    advancing_declining = np.random.rand(20)  # Placeholder for actual breadth data
    breadth = np.mean(advancing_declining)  # Mean of advancing vs. declining stocks

    features = [vol, atr, ema, breadth]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0  # Initialize reward

    # Calculate thresholds based on historical std
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -50  # Strong negative reward for BUY-aligned features
        if features[0] < 0:  # Assuming feature[0] could indicate a SELL signal
            reward += 10  # Mild positive reward for selling
    elif risk_level > 0.4:
        reward += -25  # Moderate penalty for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > historical_std:  # Uptrend and BUY signal
            reward += 20  # Positive reward for correct direction
        elif trend_direction < -0.3 and features[0] < -historical_std:  # Downtrend and SELL signal
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward