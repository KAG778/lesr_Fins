import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Exponential Moving Average (EMA) over the last 20 days
    ema = np.mean(closing_prices[-20:])  # Simple EMA for simplification

    # Feature 2: Average True Range (ATR) over the last 14 days for volatility
    high = s[3::6]  # Extract high prices
    low = s[5::6]   # Extract low prices
    true_ranges = np.maximum(high[-14:] - low[-14:], np.abs(high[-14:] - closing_prices[-15:-1]), np.abs(low[-14:] - closing_prices[-15:-1]))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0

    # Feature 3: Percentage of current price relative to the highest price in the last 20 days
    max_price = np.max(closing_prices[-20:])
    price_percentage = (closing_prices[-1] / max_price) if max_price > 0 else 0

    # Feature 4: Volatility measure based on historical standard deviation of returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0

    features = [ema, atr, price_percentage, volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract new features
    reward = 0.0  # Initialize reward

    # Calculate historical volatility for dynamic thresholds
    historical_volatility = np.std(features[3])  # Using volatility feature as a proxy
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_med = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming feature[0] indicates a BUY signal
            reward += np.random.uniform(-50, -30)  # Strong penalty
        else:
            reward += np.random.uniform(5, 10)  # Mild positive for SELL signals
    elif risk_level > risk_threshold_med:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Assuming feature[0] indicates a BUY signal
            reward += np.random.uniform(-20, -10)  # Moderate penalty

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_med:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and BUY signal
            reward += 10  # Positive reward for correct direction
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and SELL signal
            reward += 10  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Assuming feature[0] indicates a SELL signal
            reward += 5  # Reward for mean-reversion features
        else:
            reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_med:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward