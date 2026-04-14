import numpy as np

def revise_state(s):
    features = []
    
    # Price Change Feature: Percentage change from the previous closing price
    closing_price_today = s[19 * 6 + 0]  # Latest closing price
    closing_price_yesterday = s[18 * 6 + 0]  # Previous closing price
    price_change_pct = (closing_price_today - closing_price_yesterday) / closing_price_yesterday if closing_price_yesterday != 0 else 0
    features.append(price_change_pct)
    
    # Moving Average Feature: 5-day moving average of closing prices
    if len(s) >= 120:
        closing_prices = [s[i * 6 + 0] for i in range(15, 20)]  # Last 5 closing prices
        moving_average = np.mean(closing_prices) if len(closing_prices) > 0 else 0
        features.append(moving_average)

    # Volume Change Feature: Percentage change in volume from the previous day
    trading_volume_today = s[19 * 6 + 4]
    trading_volume_yesterday = s[18 * 6 + 4]
    volume_change_pct = (trading_volume_today - trading_volume_yesterday) / trading_volume_yesterday if trading_volume_yesterday != 0 else 0
    features.append(volume_change_pct)

    # Price Range Feature: Price range of the last day
    high_today = s[19 * 6 + 2]
    low_today = s[19 * 6 + 3]
    price_range_today = high_today - low_today
    features.append(price_range_today)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # Extract regime information and features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 10.0 * features[0]  # Positive reward for SELL if price change is negative
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
            reward += 15.0  # Buy signal
        elif features[0] > 0.05:  # Overbought condition
            reward += 15.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))