import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: 14-day Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 2: 14-day moving average of closing prices
    moving_average = np.mean(closing_prices[-14:])

    # Feature 3: Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                         np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if true_ranges.size > 0 else 0

    # Feature 4: Z-score of the closing price to adapt to different regimes
    mean_price = np.mean(closing_prices)
    std_price = np.std(closing_prices)
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0

    features = [rsi, moving_average, atr, z_score]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate historical std for dynamic thresholds
    historical_std = np.std(features) if features.size > 0 else 1e-6  # Avoid division by zero

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 70:  # RSI > 70 suggests overbought
            reward -= np.random.uniform(30, 50)  # Strong negative for buying
        elif features[0] < 30:  # RSI < 30 suggests oversold
            reward += np.random.uniform(5, 10)  # Mild positive for selling

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 10 * (features[0] / historical_std)  # Reward momentum
        elif trend_direction < 0:  # Downtrend
            reward += 10 * (-features[0] / historical_std)  # Reward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 30:  # Oversold condition
            reward += 10 * (features[0] / historical_std)
        elif features[0] > 70:  # Overbought condition
            reward -= 10 * (features[0] / historical_std)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds