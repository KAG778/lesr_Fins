import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices

    # Feature 1: 14-day Exponential Moving Average (EMA)
    ema_period = 14
    if len(closing_prices) >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])  # Simplified for illustrative purposes
    else:
        ema = np.nan  

    # Feature 2: Average True Range (ATR) for volatility measurement
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.nan

    # Feature 3: Rate of Change (ROC)
    if len(closing_prices) >= 14:
        roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] * 100
    else:
        roc = np.nan
    
    # Feature 4: Z-score of recent returns for mean reversion
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100 if len(closing_prices) > 1 else np.array([np.nan])
    z_score = (daily_returns[-1] - np.mean(daily_returns[-14:])) / np.std(daily_returns[-14:]) if len(daily_returns) >= 14 else np.nan
    
    # Return only the computed features, filtering out NaN values
    features = [ema, atr, roc, z_score]
    
    # Ensure all features are valid numbers (replace NaN with 0)
    features = [f if np.isfinite(f) else 0 for f in features]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(10, 20)   # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if risk_level < 0.4:
        if abs(trend_direction) > 0.3:
            reward += (10 * trend_direction)  # Amplify reward based on trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        z_score = enhanced_s[123][3]  # Using the z-score from revised features
        if z_score < -1:  # Indicates oversold condition
            reward += 15  # Reward for potential buy
        elif z_score > 1:  # Indicates overbought condition
            reward -= 15  # Penalize for potential buy

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range