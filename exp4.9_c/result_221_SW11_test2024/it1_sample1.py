import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes
    
    features = []
    
    # Feature 1: Crisis Detection (Standard Deviation of Returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    crisis_detection = np.std(returns)  # Higher std indicates more risk
    features.append(crisis_detection)

    # Feature 2: Trend Strength (5-day moving average of price change)
    if len(closing_prices) >= 5:
        trend_strength = np.mean(np.diff(closing_prices[-5:]) / closing_prices[-6:-1])
    else:
        trend_strength = 0.0
    features.append(trend_strength)

    # Feature 3: Bollinger Bands (percentage from the moving average)
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        moving_std = np.std(closing_prices[-20:])
        upper_band = moving_avg + (2 * moving_std)
        lower_band = moving_avg - (2 * moving_std)
        last_price = closing_prices[-1]
        if upper_band != lower_band:
            bollinger_position = (last_price - lower_band) / (upper_band - lower_band)
        else:
            bollinger_position = 0.0  # Prevent division by zero
    else:
        bollinger_position = 0.0
    features.append(bollinger_position)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate relative thresholds using historical data
    historical_std = np.std(features)  # std of the new features as a risk measure
    volatility_threshold = 1.5 * historical_std  # Adjusted threshold for high volatility

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > volatility_threshold:  # High crisis detection
            reward = -np.random.uniform(30, 50)  # Strong negative reward for BUY
        else:
            reward = np.random.uniform(5, 10)  # Mild positive for SELL
        return np.clip(reward, -100, 100)

    if risk_level > 0.4:
        reward = -10  # Moderate negative reward for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[1] > 0:  # Uptrend aligned with trend strength
            reward += 20
        elif trend_direction < -0.3 and features[1] < 0:  # Downtrend aligned with trend strength
            reward += 20

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0.1:  # Assuming Bollinger Band position indicates oversold
            reward += 15  # Reward for buying in oversold condition
        elif features[2] > 0.9:  # Assuming Bollinger Band position indicates overbought
            reward += 15  # Reward for selling in overbought condition

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds