import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
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

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= 40.0 if features[0] > 0 else 0  # Positive price change suggests buy
        reward -= 10.0 if features[0] <= 0 else 0  # Negative price change suggests sell
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 10.0 if features[0] > 0 else 0  # Positive price change suggests buy

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive price change
            reward += 10.0 * trend_direction  # Reward for aligning with trend
        else:  # Negative price change
            reward += 10.0 * (-trend_direction)  # Reward for aligning with bearish trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Oversold condition
            reward += 5.0  # Reward for considering buy
        elif features[0] > 0.05:  # Overbought condition
            reward += 5.0  # Reward for considering sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))