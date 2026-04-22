import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Moving Average (last 14 days)
    if len(closing_prices) >= 14:
        moving_average = np.mean(closing_prices[-14:])
    else:
        moving_average = closing_prices[-1] if len(closing_prices) > 0 else 0

    # Feature 2: Price Momentum (percentage change over last 5 days)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100 if len(closing_prices) >= 6 else 0

    # Feature 3: Volume Change (percentage change from previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 if len(volumes) >= 2 and volumes[-2] > 0 else 0

    # Feature 4: Volatility (standard deviation of the last 14 days' returns)
    if len(closing_prices) >= 14:
        returns = np.diff(closing_prices[-14:]) / closing_prices[-14:-1]
        volatility = np.std(returns) * 100  # Convert to percentage
    else:
        volatility = 0  # Default to 0 if not enough data

    # Feature 5: Relative Strength Index (RSI)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if not enough data

    features = [moving_average, price_momentum, volume_change, volatility, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Dynamic thresholds using historical data (placeholder for standard deviation calculations)
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    trend_threshold_high = 0.3
    trend_threshold_low = -0.3
    mean_reversion_threshold = 50  # Example for RSI

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)   # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold_high and risk_level < risk_threshold_medium:
        if trend_direction > trend_threshold_high:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < trend_threshold_low:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold_high and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]