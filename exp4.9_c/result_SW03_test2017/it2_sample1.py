import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0:120:6]  # Every 6th element starting from index 0 is the closing price
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Exponential Moving Average (EMA) - 20 days
    if len(closing_prices) >= 20:
        weights = np.exp(np.linspace(-1, 0, 20))
        weights /= weights.sum()
        ema = np.dot(weights, closing_prices[-20:])
    else:
        ema = 0.0
    features.append(ema)

    # Feature 2: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.mean(delta[delta > 0]) if len(delta[delta > 0]) > 0 else 0
    loss = -np.mean(delta[delta < 0]) if len(delta[delta < 0]) > 0 else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 3: Average True Range (ATR) - 14 days
    if len(closing_prices) >= 14:
        high_prices = s[2:120:6]
        low_prices = s[3:120:6]
        tr = np.maximum(high_prices[-14:] - low_prices[-14:], 
                        np.maximum(np.abs(high_prices[-14:] - closing_prices[-15:-1]), 
                                   np.abs(low_prices[-14:] - closing_prices[-15:-1])))
        atr = np.mean(tr)
    else:
        atr = 0.0
    features.append(atr)

    # Feature 4: Z-score of current price relative to historical mean
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:]) if np.std(closing_prices[-20:]) > 0 else 1  # Avoid division by zero
        z_score = (closing_prices[-1] - mean_price) / std_dev
    else:
        z_score = 0.0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 15)    # Mild positive for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 20 * abs(trend_direction)  # Reward momentum alignment based on strength of trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        rsi = enhanced_s[123][1]  # Assuming RSI is the second feature
        if rsi < 30:  # Oversold
            reward += 15.0  # Reward for mean-reversion BUY
        elif rsi > 70:  # Overbought
            reward += 15.0  # Reward for mean-reversion SELL

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within specified range