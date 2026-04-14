import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Price Change Percentage (last day vs previous day)
    closing_prices = s[0::6]
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_change_pct)

    # Feature 2: Average Volume Change (5-day average of the last 5 days)
    volumes = s[4::6]
    avg_volume_change = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 0
    features.append(avg_volume_change)

    # Feature 3: Price Range (High - Low) of the most recent day
    high_prices = s[2::6]
    low_prices = s[3::6]
    price_range = high_prices[-1] - low_prices[-1]
    features.append(price_range)

    # Feature 4: Volatility (standard deviation of the last 5 closing prices)
    if len(closing_prices) >= 6:
        volatility = np.std(closing_prices[-5:])
    else:
        volatility = 0
    features.append(volatility)

    # Feature 5: Relative Strength Index (RSI) - Overbought/Oversold condition
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

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 7.0 * features[0]  # Mild positive for SELL-aligned features based on price change
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Positive reward for upward price momentum
        else:  # Downtrend
            reward += -features[0] * 20.0  # Negative reward for downward price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # RSI Oversold condition
            reward += 15.0  # Reward for potential buy signal
        elif features[3] > 70:  # RSI Overbought condition
            reward += 15.0  # Reward for potential sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))