import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Momentum (current close - close 5 days ago) normalized by current price
    price_momentum = (s[6*19] - s[6*14]) / s[6*19] if s[6*19] != 0 else 0
    features.append(price_momentum)
    
    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(s[4::6])  # every 6th entry starting from index 4
    features.append(avg_volume)
    
    # Feature 3: Price Range (max high - min low) normalized by average price
    high_prices = s[2::6]
    low_prices = s[3::6]
    price_range = (np.max(high_prices) - np.min(low_prices)) / np.mean(s[0::6]) if np.mean(s[0::6]) != 0 else 0
    features.append(price_range)
    
    # Feature 4: Relative Strength Index (RSI) for last 14 days
    closing_prices = s[0::6]
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean() if np.any(delta > 0) else 0
    loss = -np.where(delta < 0, delta, 0).mean() if np.any(delta < 0) else 0
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)
    
    # Feature 5: Standard Deviation of Daily Returns (Volatility)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if len(returns) > 0 else 0
    features.append(volatility)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0
    
    # Calculate historical std dev for relative thresholds
    historical_volatility = np.std(features[4])  # Assuming features[4] is our volatility measure
    relative_risk_threshold = 0.4 * historical_volatility  # Adjust based on historical data

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -50 if features[0] > 0 else 10  # Strong negative for BUY-aligned features, mild positive for SELL
        return np.clip(reward, -100, 100)
    
    elif risk_level > 0.4:
        reward += -20 if features[0] > 0 else 0  # Moderate negative reward for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if (trend_direction > 0 and features[0] > 0) or (trend_direction < 0 and features[0] < 0):
            reward += 20  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # Oversold condition for RSI
            reward += 15  # Encourage buying
        elif features[3] > 70:  # Overbought condition for RSI
            reward += 15  # Encourage selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the bounds