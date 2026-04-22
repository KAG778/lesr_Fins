import numpy as np

def revise_state(s):
    # Extract relevant data from the state
    closing_prices = s[0::6][:20]  # Extract closing prices
    volumes = s[4::6][:20]          # Extract trading volumes
    high_prices = s[2::6][:20]      # Extract high prices
    low_prices = s[3::6][:20]       # Extract low prices

    # Feature 1: Volatility - standard deviation of returns over the last 10 days
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0

    # Feature 2: Crisis detection - percentage drawdown from recent peak
    peak_price = np.max(closing_prices[-20:])  # Peak over the last 20 days
    drawdown = (peak_price - closing_prices[-1]) / peak_price if peak_price > 0 else 0

    # Feature 3: Momentum - price momentum adjusted for volatility
    momentum = (closing_prices[-1] - closing_prices[-6]) / (volatility + 1e-10)

    # Feature 4: Choppiness index - measure of market trend vs. chop
    if len(returns) >= 14:
        high_low_range = np.max(high_prices[-14:]) - np.min(low_prices[-14:])
        choppiness_index = (np.log(high_low_range) / np.log(np.max(closing_prices[-14:]) - np.min(closing_prices[-14:]))) * 100
    else:
        choppiness_index = 100  # Default high chop if insufficient data

    features = [volatility, drawdown, momentum, choppiness_index]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Calculate historical thresholds based on past data
    historical_std = np.std(features) if np.std(features) != 0 else 1e-10
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        if features[2] > 0:  # If momentum is positive
            reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        else:
            reward += np.random.uniform(5, 10)  # Mild positive for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > 0 and features[2] > 0:  # Bullish trend and bullish momentum
            reward += 20  # Strong positive reward
        elif trend_direction < 0 and features[2] < 0:  # Bearish trend and bearish momentum
            reward += 20  # Strong positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[1] > 0.1:  # Drawdown indicates potential for mean reversion
            reward += 15  # Positive reward for potential buy
        else:
            reward -= 10  # Penalize for breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)