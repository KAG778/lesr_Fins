import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extracting closing prices
    n = len(closing_prices)

    # Feature 1: 14-day Exponential Moving Average (EMA)
    if n >= 14:
        ema = np.zeros(n)
        ema[:14] = np.nan  # Fill initial values with NaN
        ema[14-1] = np.mean(closing_prices[:14])  # Calculate the initial EMA
        for i in range(14, n):
            alpha = 2 / (14 + 1)
            ema[i] = (closing_prices[i] - ema[i-1]) * alpha + ema[i-1]
    else:
        ema = np.nan * np.ones(n)

    # Feature 2: Average True Range (ATR) for volatility measure (20-day)
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.zeros(n)
    if n >= 20:
        for i in range(19, n):
            atr[i] = np.mean(tr[i-19:i+1])
    else:
        atr = np.nan * np.ones(n)

    # Feature 3: 5-day Mean Reversion Indicator
    mean_reversion = np.zeros(n)
    for i in range(5, n):
        mean_reversion[i] = np.mean(closing_prices[i-5:i]) - closing_prices[i]

    # Return features: last value of EMA, ATR, and mean_reversion
    features = [ema[-1] if n >= 14 else 0, 
                atr[-1] if n >= 20 else 0, 
                mean_reversion[-1] if n >= 5 else 0]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    # Initialize reward
    reward = 0.0

    # Calculate thresholds based on historical volatility
    historical_std = np.std(features) if len(features) > 0 else 1.0
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std

    # Priority 1: Risk Management
    if risk_level > high_risk_threshold:
        if features[0] > 0:  # BUY-aligned feature (e.g., EMA)
            reward = -40  # Strong negative for risky BUY
        else:
            reward = 10  # Mild positive for SELL
    elif risk_level > medium_risk_threshold:
        if features[0] > 0:  # BUY signal
            reward = -10  # Moderate negative for risky BUY

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend with positive momentum
            reward += 20  # Positive reward
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend with negative momentum
            reward += 20  # Positive reward

    # Priority 3: Sideways/Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0:  # Oversold situation
            reward += 15  # Positive for mean-reversion buying
        elif features[2] > 0:  # Overbought situation
            reward += 15  # Positive for mean-reversion selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 * historical_std and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range