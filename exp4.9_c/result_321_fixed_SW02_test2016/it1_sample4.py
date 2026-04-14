import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Price Change Percentage (last day vs previous day)
    closing_prices = s[0::6]
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_change_pct)

    # Feature 2: 5-day Moving Average (to capture trends)
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    features.append(moving_average)

    # Feature 3: Volume Change Percentage (today vs yesterday)
    volumes = s[4::6]
    volume_change_pct = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
    features.append(volume_change_pct)

    # Feature 4: Price Range (High - Low) of the last day
    high_prices = s[2::6]
    low_prices = s[3::6]
    price_range = high_prices[-1] - low_prices[-1] if high_prices[-1] != low_prices[-1] else 0
    features.append(price_range)

    # Feature 5: Volatility (standard deviation of the last 5 closing prices)
    if len(closing_prices) >= 5:
        volatility = np.std(closing_prices[-5:])
    else:
        volatility = 0
    features.append(volatility)

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
        reward += 10.0 * (1 if features[0] < 0 else 0)  # Mild positive for SELL-aligned features (if price change suggests sell)
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
        if features[0] < -0.05:  # Oversold condition
            reward += 10.0  # Buy signal
        elif features[0] > 0.05:  # Overbought condition
            reward += 10.0  # Sell signal
        else:
            reward -= 5.0  # Penalize for breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))