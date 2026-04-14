import numpy as np

def revise_state(s):
    features = []
    
    # Price Change Feature
    closing_price_today = s[19 * 6 + 0]  # Closing price of the most recent day
    opening_price_today = s[19 * 6 + 1]   # Opening price of the most recent day
    price_change = (closing_price_today - opening_price_today) / opening_price_today if opening_price_today != 0 else 0
    features.append(price_change)

    # Moving Average Feature (5-day moving average)
    if len(s) >= 120:  # Ensure we have enough data for moving average
        closing_prices = [s[i * 6 + 0] for i in range(15, 20)]  # Last 5 closing prices
        moving_average = np.mean(closing_prices) if len(closing_prices) > 0 else 0
        features.append(moving_average)

    # Volume Change Feature
    trading_volume_today = s[19 * 6 + 4]  # Volume of the most recent day
    trading_volume_yesterday = s[18 * 6 + 4]  # Volume of the previous day
    volume_change = (trading_volume_today - trading_volume_yesterday) / trading_volume_yesterday if trading_volume_yesterday != 0 else 0
    features.append(volume_change)
    
    # Standard Deviation of Closing Prices for crisis detection
    closing_prices_last_20 = [s[i * 6 + 0] for i in range(0, 20)]  # Last 20 closing prices
    if len(closing_prices_last_20) > 0:
        std_dev = np.std(closing_prices_last_20)
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
    std_dev = features[3]  # Standard deviation of closing prices as crisis indicator

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
        if features[0] < -0.05:  # Oversold condition
            reward += 15.0  # Buy signal
        elif features[0] > 0.05:  # Overbought condition
            reward += 15.0  # Sell signal
        else:
            reward -= 5.0  # Penalize for breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    # Implementing a crisis penalty based on standard deviation
    if std_dev > np.mean(features[3:]) * 1.5:  # Arbitrarily chosen threshold for crisis
        reward -= 20.0  # Additional penalty during crisis periods

    return float(np.clip(reward, -100, 100))