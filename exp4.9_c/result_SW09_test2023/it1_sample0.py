import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: 5-day Moving Average of Closing Prices
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    features.append(moving_average)

    # Feature 2: 5-day Price Momentum
    price_momentum = closing_prices[-1] - closing_prices[-5]  # Current price - price 5 days ago
    features.append(price_momentum)

    # Feature 3: Volatility Measure (Standard Deviation of Closing Prices over last 5 days)
    volatility = np.std(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    features.append(volatility)

    # Feature 4: Crisis Indicator (Percentage drop from the peak in the last 20 days)
    peak_price = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    crisis_indicator = (peak_price - closing_prices[-1]) / peak_price if peak_price > 0 else 0
    features.append(crisis_indicator)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Historical thresholds (for the sake of example, let's say we have pre-computed them)
    risk_threshold_high = 0.2  # Example value, should be based on historical std deviation
    risk_threshold_medium = 0.1  # Example value

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative for BUY-aligned features
        reward += 5   # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Strong positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Strong positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.1:
        reward += 10  # Reward for mean-reversion features
        reward -= 5   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))

    return reward