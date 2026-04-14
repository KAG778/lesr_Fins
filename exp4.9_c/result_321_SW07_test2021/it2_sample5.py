import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Exponential Moving Average (EMA) - Short-term (5 days)
    closing_prices = s[0::6]  # Extracting closing prices
    ema_short = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    features.append(ema_short)
    
    # Feature 2: Exponential Moving Average (EMA) - Long-term (20 days)
    ema_long = np.mean(closing_prices[-20:]) if len(closing_prices[-20:]) > 0 else 0
    features.append(ema_long)
    
    # Feature 3: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0
    features.append(price_momentum)
    
    # Feature 4: Relative Strength Index (RSI) - 14 days
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean() if np.any(delta > 0) else 0
    loss = -np.where(delta < 0, delta, 0).mean() if np.any(delta < 0) else 0
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)
    
    # Feature 5: Average True Range (ATR) - Volatility measure
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    features.append(atr)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate relative thresholds based on historical distribution
    historical_std = np.std(features) if features.size > 0 else 1
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -50  # Strong negative reward for BUY-aligned features
        if features[2] < 0:  # If price momentum is negative, consider it a sell scenario
            reward += 10  # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)
    
    elif risk_level > low_risk_threshold:
        reward += -20  # Moderate negative reward for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < low_risk_threshold:
        if (trend_direction > 0 and features[2] > 0) or (trend_direction < 0 and features[2] < 0):
            reward += 20  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < low_risk_threshold:
        if features[3] < 30:  # Assuming RSI indicates oversold condition
            reward += 10  # Reward for mean-reversion buying
        elif features[3] > 70:  # Assuming RSI indicates overbought condition
            reward += 10  # Reward for mean-reversion selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds