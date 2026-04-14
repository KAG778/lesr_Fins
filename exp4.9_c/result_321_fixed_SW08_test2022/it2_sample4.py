import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices

    # 1. Price Change Percentage (last day)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # 2. Average Volume
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # 3. Historical Volatility (standard deviation of closing prices)
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 0 else 0.0

    # 4. Price Range (high - low)
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 and len(low_prices) > 0 else 0.0

    # 5. 5-Day Moving Average
    moving_average_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    closing_vs_moving_avg = closing_prices[-1] - moving_average_5

    # Return the features as a numpy array
    features = [price_change_pct, average_volume, historical_volatility, price_range, closing_vs_moving_avg]
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
        reward += 10.0 * (1 - features[0])  # Mild positive for SELL-aligned features (if price change is negative)
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Reward for positive price change alignment
        else:  # Downtrend
            reward += -features[0] * 20.0  # Reward for negative price change alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition (price change)
            reward += 15.0  # Encourage buying
        elif features[0] > 0.01:  # Overbought condition (price change)
            reward += -15.0  # Encourage selling to avoid chasing breakouts

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))