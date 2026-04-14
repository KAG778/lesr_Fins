import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]  # Extract low prices
    volumes = s[4::6]  # Extract trading volumes

    features = []

    # Feature 1: Price Change Percentage
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_change)

    # Feature 2: Moving Average (last 5 days)
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = np.mean(closing_prices)  # Fallback if less than 5 days
    features.append(moving_average)

    # Feature 3: Bollinger Bands (20-day period)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        features.append((closing_prices[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0)
    else:
        features.append(0)  # Fallback if less than 20 days

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward = -40.0  # Example strong negative reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -20.0  # Example moderate negative reward

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Reward for positive trend
            reward += 10.0  # Example positive reward for a BUY signal
        elif trend_direction < -0.3:
            # Reward for negative trend
            reward += 10.0  # Example positive reward for a SELL signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += 5.0  # Example positive reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)