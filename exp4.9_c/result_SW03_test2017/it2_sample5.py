import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Every 6th element starting from index 0 is the closing price
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Rate of Change over 14 days (Momentum)
    if closing_prices[-14] != 0:
        momentum = (closing_prices[-1] - closing_prices[-14]) / closing_prices[-14]
    else:
        momentum = 0.0
    features.append(momentum)

    # Feature 2: Historical Volatility (20-day)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(20) if len(daily_returns) > 0 else 0.0
    features.append(historical_volatility)

    # Feature 3: Average True Range (ATR) over the last 14 days
    if len(closing_prices) >= 14:
        high_prices = s[2:120:6]
        low_prices = s[3:120:6]
        tr = np.maximum(high_prices[-14:] - low_prices[-14:],
                        np.maximum(np.abs(high_prices[-14:] - closing_prices[-15:-1]),
                                   np.abs(low_prices[-14:] - closing_prices[-15:-1])))
        atr = np.mean(tr) if len(tr) > 0 else 0
    else:
        atr = 0
    features.append(atr)

    # Feature 4: Z-score of current price relative to the historical mean (last 20 days)
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:]) if np.std(closing_prices[-20:]) > 0 else 1  # Avoid division by zero
        z_score = (closing_prices[-1] - mean_price) / std_dev
    else:
        z_score = 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical features
    historical_std = np.std(enhanced_s[123:])
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50 * (risk_level - high_risk_threshold)  # Strong negative for BUY-aligned features
        if trend_direction < 0:
            reward += 15 * (1 - risk_level)  # Mild positive for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= 20 * (risk_level - low_risk_threshold)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 25 * abs(trend_direction)  # Reward for momentum alignment based on trend strength

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward stays within range