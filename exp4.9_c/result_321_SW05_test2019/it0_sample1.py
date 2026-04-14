import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate moving averages for the last 5 and 10 days
    closing_prices = s[::6][:20]  # Extract closing prices from the raw state
    if len(closing_prices) >= 10:
        ma5 = np.mean(closing_prices[-5:])  # 5-day moving average
        ma10 = np.mean(closing_prices[-10:])  # 10-day moving average
        features.append(ma5)
        features.append(ma10)
    else:
        features.append(np.nan)  # Handle edge case
        features.append(np.nan)

    # Calculate Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    average_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    average_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = average_gain / average_loss if average_loss > 0 else 0  # Avoid division by zero

    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Calculate Price Change Percentage
    if closing_prices[-1] != 0:
        price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    else:
        price_change_pct = 0  # Handle edge case

    features.append(price_change_pct)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strongly negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Random strong negative reward for BUY
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)  # Random moderate negative reward for BUY
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for BUY-aligned features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for SELL-aligned features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += 10  # Positive reward for mean-reversion signals

    # Penalize breakout-chasing features (not implemented in detail here)
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward