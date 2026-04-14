import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Price Change Percentage (today vs yesterday)
    closing_prices = s[0::6]
    if len(closing_prices) >= 2:
        price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # Feature 2: 5-day Moving Average for price to capture trends
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = 0
    features.append(moving_average)

    # Feature 3: Volume Change Percentage (today vs yesterday)
    volumes = s[4::6]
    if len(volumes) >= 2:
        volume_change_pct = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
    else:
        volume_change_pct = 0
    features.append(volume_change_pct)

    # Feature 4: Price Range (High - Low) of the last day
    high_prices = s[2::6]
    low_prices = s[3::6]
    if len(high_prices) > 0 and len(low_prices) > 0:
        price_range = high_prices[-1] - low_prices[-1] if high_prices[-1] != low_prices[-1] else 0
    else:
        price_range = 0
    features.append(price_range)

    # Feature 5: Volatility (standard deviation of the last 5 closing prices)
    if len(closing_prices) >= 5:
        volatility = np.std(closing_prices[-5:])
    else:
        volatility = 0
    features.append(volatility)

    # Feature 6: Relative Strength Index (RSI) - Overbought/Oversold
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI when not enough data
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Calculate historical standard deviation for thresholding
    historical_std = np.std(features[4:])  # Using volatility feature for relative thresholds

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        if features[0] < 0:  # If price change is negative, mild positive for SELL
            reward += 10.0  # Positive reward for selling during high risk

    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Strong reward for favorable price change
        else:  # Downtrend
            reward += -features[0] * 20.0  # Strong reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[5] < 30:  # RSI < 30 indicates oversold
            reward += 15.0  # Reward for considering buy
        elif features[5] > 70:  # RSI > 70 indicates overbought
            reward += 15.0  # Reward for considering sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    # Adjusting reward for crisis periods based on volatility
    if volatility_level > historical_std * 1.5:  # Arbitrary threshold for crisis periods
        reward -= 20.0  # Additional penalty during crisis

    return float(np.clip(reward, -100, 100))