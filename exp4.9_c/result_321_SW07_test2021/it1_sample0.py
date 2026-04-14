import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Exponential Moving Average (EMA) - Short-term (5 days)
    ema_short = np.mean(s[0::6][-5:]) if len(s[0::6][-5:]) > 0 else 0
    features.append(ema_short)
    
    # Feature 2: Exponential Moving Average (EMA) - Long-term (20 days)
    ema_long = np.mean(s[0::6][-20:]) if len(s[0::6][-20:]) > 0 else 0
    features.append(ema_long)
    
    # Feature 3: Price Range (High - Low) over the last 20 days
    price_high = np.max(s[2::6][-20:]) if len(s[2::6][-20:]) > 0 else 0
    price_low = np.min(s[3::6][-20:]) if len(s[3::6][-20:]) > 0 else 0
    price_range = price_high - price_low
    features.append(price_range)
    
    # Feature 4: Average True Range (ATR) for volatility
    true_ranges = np.abs(np.diff(s[2::6][-20:]))  # Highs and Lows for ATR calculation
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    features.append(atr)
    
    # Feature 5: Modified Relative Strength Index (RSI) - using a longer period (14)
    delta = np.diff(s[0::6])
    gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Calculate historical standard deviation for dynamic thresholds
    historical_std = np.std(features)
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -50  # Strong negative for BUY-aligned features
        if features[0] < 0:  # If EMA short is below EMA long, consider it a sell scenario
            reward += 10  # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)
    
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals
        return np.clip(reward, -100, 100)
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > features[1]:  # Uptrend with positive momentum
            reward += 20
        elif trend_direction < 0 and features[0] < features[1]:  # Downtrend with negative momentum
            reward += 20
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[4] < 30:  # Oversold condition for RSI
            reward += 15  # Encourage buying
        elif features[4] > 70:  # Overbought condition for RSI
            reward += 15  # Encourage selling
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility conditions
    
    return np.clip(reward, -100, 100)  # Ensure reward stays in [-100, 100]