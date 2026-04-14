import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices
    closing_prices = s[0:120:6]  # Every 6th element starting from index 0 is the closing price
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Recent Price Momentum (Rate of Change)
    if closing_prices[-2] != 0:
        momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        momentum = 0.0
    features.append(momentum)

    # Feature 2: Historical Volatility (20-day)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(20) if len(daily_returns) > 0 else 0.0
    features.append(historical_volatility)

    # Feature 3: Average Trading Volume (last 20 days)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes) if len(volumes) > 0 else 0
    features.append(avg_volume)

    # Feature 4: Price Range (High - Low) of the last 5 days
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    if len(high_prices) >= 5 and len(low_prices) >= 5:
        price_range = np.mean(high_prices[-5:]) - np.mean(low_prices[-5:])
    else:
        price_range = 0.0
    features.append(price_range)

    # Feature 5: Momentum Relative to Historical Volatility
    if historical_volatility != 0:
        momentum_to_volatility = momentum / historical_volatility
    else:
        momentum_to_volatility = 0.0
    features.append(momentum_to_volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Dynamic thresholds based on historical volatility
    high_risk_threshold = 0.7 * np.std(enhanced_s[123:])  # Using historical standard deviation
    low_risk_threshold = 0.4 * np.std(enhanced_s[123:])

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 15)    # Mild positive for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:  # Uptrend
            reward += 15  # Reward for upward momentum
        elif trend_direction < 0:  # Downtrend
            reward += 15  # Reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward stays within range