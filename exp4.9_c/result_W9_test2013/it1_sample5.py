import numpy as np

def revise_state(s):
    features = []

    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Price Change Percentage over the last 10 days
    if len(closing_prices) >= 10:
        price_change_pct = (closing_prices[-1] - closing_prices[-10]) / closing_prices[-10] if closing_prices[-10] != 0 else 0
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # Feature 2: Average Volume Change Percentage over the last 10 days
    if len(volumes) >= 10:
        avg_volume_change = (volumes[-1] - np.mean(volumes[-10:])) / np.mean(volumes[-10:]) if np.mean(volumes[-10:]) != 0 else 0
    else:
        avg_volume_change = 0
    features.append(avg_volume_change)

    # Feature 3: Bollinger Bands (Upper Band) based on last 20 days
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (rolling_std * 2)  # 2 standard deviations
    else:
        upper_band = closing_prices[-1] if len(closing_prices) > 0 else 0
    features.append(upper_band)

    # Feature 4: Price Momentum (current - previous day)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) >= 2 else 0
    features.append(price_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # The new features added
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # If price change percentage is positive (indicating bullish)
            reward -= np.random.uniform(30, 50)  # Strong negative for BUY
        else:
            reward += np.random.uniform(5, 10)  # Mild positive for SELL
    elif risk_level > 0.4:
        if features[0] > 0:
            reward -= 15  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish
            reward += 20  # Positive reward for correct bullish signal
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish
            reward += 20  # Positive reward for correct bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Assuming negative features indicate selling
            reward += 15  # Reward for mean-reversion behavior

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(np.clip(reward, -100, 100))