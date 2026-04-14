import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Price Change Percentage (last day vs previous day)
    closing_prices = s[0::6]
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_change_pct)

    # Feature 2: 10-day Moving Average (to capture longer-term trends)
    if len(closing_prices) >= 10:
        moving_average_10 = np.mean(closing_prices[-10:])
    else:
        moving_average_10 = 0
    features.append(moving_average_10)

    # Feature 3: Average Volume Change (10-day average)
    volumes = s[4::6]
    avg_volume_change = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
    features.append(avg_volume_change)

    # Feature 4: Price Range (High - Low) of the last day
    high_prices = s[2::6]
    low_prices = s[3::6]
    price_range = high_prices[-1] - low_prices[-1] if high_prices[-1] != low_prices[-1] else 0
    features.append(price_range)

    # Feature 5: Standard Deviation of Closing Prices (last 20 days) for volatility
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
    price_change_pct = features[0]  # Price change percentage
    std_dev = features[4]  # Standard deviation for crisis detection

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        if price_change_pct < 0:  # Positive reward for SELL if price change is negative
            reward += 10.0
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += price_change_pct * 20.0  # Reward for aligning with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Use historical std to determine overbought/oversold conditions
        if price_change_pct < -0.05 * std_dev:  # Oversold condition
            reward += 15.0  # Reward for potential buy signal
        elif price_change_pct > 0.05 * std_dev:  # Overbought condition
            reward += 15.0  # Reward for potential sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))