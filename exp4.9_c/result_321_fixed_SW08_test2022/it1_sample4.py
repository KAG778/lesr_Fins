import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Extract every 6th element starting from index 0 (closing prices)
    volumes = s[4:120:6]         # Extract every 6th element starting from index 4 (volumes)

    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: 20-Day Historical Volatility
    historical_volatility = np.std(closing_prices)

    # Feature 4: Momentum (Current Price vs. 5-Day Moving Average)
    moving_average_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    momentum = closing_prices[-1] - moving_average_5

    # Feature 5: Price Range (High - Low)
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 else 0

    features = [price_change_pct, average_volume, historical_volatility, momentum, price_range]
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
        reward -= 50.0  # Strong negative for BUY-aligned features
        reward += 10.0 * (1 - features[0])  # Mild positive reward for SELL-aligned features (if price change is negative)
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 30.0  # Reward based on price change
        else:  # Downtrend
            reward += -features[0] * 30.0  # Reward based on negative price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.02:  # Oversold condition
            reward += 15.0  # Encourage buying
        elif features[0] > 0.02:  # Overbought condition
            reward += -10.0  # Encourage selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.75  # Reduce reward magnitude by 25%

    return float(np.clip(reward, -100, 100))