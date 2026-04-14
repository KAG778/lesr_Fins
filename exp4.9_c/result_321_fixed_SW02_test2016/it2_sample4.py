import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Change Percentage (last day vs previous day)
    closing_prices = s[0::6]
    if len(closing_prices) >= 2:
        price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # Feature 2: 5-day Moving Average Change (to capture short-term trends)
    if len(closing_prices) >= 6:
        moving_average_change = np.mean(closing_prices[-5:]) - closing_prices[-1]
    else:
        moving_average_change = 0
    features.append(moving_average_change)

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
    if len(high_prices) >= 1 and len(low_prices) >= 1:
        price_range = high_prices[-1] - low_prices[-1]
    else:
        price_range = 0
    features.append(price_range)

    # Feature 5: Historical Volatility (standard deviation of closing prices over the last 20 days)
    if len(closing_prices) >= 20:
        historical_volatility = np.std(closing_prices[-20:])
    else:
        historical_volatility = 0
    features.append(historical_volatility)

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
        reward += 10.0 * (1 if features[0] < 0 else 0)  # Mild positive for SELL-aligned features if price change is negative
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
        if features[0] < -0.05:  # Oversold condition
            reward += 15.0  # Reward for potential buy signal
        elif features[0] > 0.05:  # Overbought condition
            reward += 15.0  # Reward for potential sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))