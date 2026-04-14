import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Price Change Percentage (last day vs previous day)
    closing_prices = s[0::6]
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_change_pct)

    # Feature 2: Average Volume Change (10-day average of the last 10 days)
    volumes = s[4::6]
    avg_volume_change = np.mean(volumes[-10:]) if len(volumes[-10:]) > 0 else 0
    features.append(avg_volume_change)

    # Feature 3: Price Range (High - Low) of the last day
    high_prices = s[2::6]
    low_prices = s[3::6]
    price_range = high_prices[-1] - low_prices[-1] if high_prices[-1] != low_prices[-1] else 0
    features.append(price_range)

    # Feature 4: 14-day Relative Strength Index (RSI) to gauge overbought/oversold conditions
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Default RSI if not enough data
    features.append(rsi)

    # Feature 5: Standard Deviation of Closing Prices for crisis detection
    if len(closing_prices) >= 20:
        std_dev = np.std(closing_prices[-20:])
    else:
        std_dev = 0
    features.append(std_dev)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 10.0 * (1 if features[0] < 0 else 0)  # Mild positive for SELL if price change suggests sell
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Positive reward for upward momentum
        else:  # Downtrend
            reward += -features[0] * 20.0  # Positive reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # RSI < 30 indicates oversold
            reward += 15.0  # Reward for considering buy
        elif features[3] > 70:  # RSI > 70 indicates overbought
            reward += 15.0  # Reward for considering sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    # Implementing a crisis penalty based on standard deviation
    if features[4] > np.mean(features[4:]) * 1.5:  # Arbitrarily chosen threshold for crisis
        reward -= 20.0  # Additional penalty during crisis periods

    return float(np.clip(reward, -100, 100))