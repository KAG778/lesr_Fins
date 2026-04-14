import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    num_days = len(closing_prices)
    
    # Feature 1: 14-day Exponential Moving Average (EMA) for trend detection
    if num_days >= 14:
        ema = np.mean(closing_prices[-14:])  # Simplified EMA calculation for illustration
    else:
        ema = 0  # Use 0 for invalid cases
    
    # Feature 2: Average True Range (ATR) for volatility measurement
    if num_days >= 14:
        highs = s[2::6]
        lows = s[3::6]
        close_prev = closing_prices[-2] if num_days > 1 else closing_prices[0]
        tr = np.maximum(highs[-14:] - lows[-14:], 
                        np.maximum(np.abs(highs[-14:] - close_prev), 
                                   np.abs(lows[-14:] - close_prev)))
        atr = np.mean(tr)
    else:
        atr = 0  # Use 0 for invalid cases
    
    # Feature 3: Z-score of the last 14-day returns for mean reversion detection
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    if len(daily_returns) >= 14:
        mean_return = np.mean(daily_returns[-14:])
        std_return = np.std(daily_returns[-14:])
        z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    else:
        z_score = 0  # Use 0 for invalid cases
    
    # Feature 4: 14-day Relative Strength Index (RSI) for momentum
    deltas = np.diff(closing_prices)  # Daily price changes
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # Compile features
    features = [ema, atr, z_score, rsi]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate relative thresholds based on historical volatility
    historical_volatility = np.std(enhanced_s[123:])  # Use computed features for historical volatility
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_mid = 0.4 * historical_volatility
    trend_threshold_high = 0.3 * historical_volatility
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for high risk
        reward += 10   # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if risk_level < risk_threshold_mid:
        if abs(trend_direction) > trend_threshold_high:
            reward += 20 * np.sign(trend_direction)  # Positive reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold_high and risk_level < risk_threshold_mid:
        z_score = enhanced_s[123][2]  # Using z-score for mean reversion
        if z_score < -1:  # Oversold condition
            reward += 15  # Reward for potential buy
        elif z_score > 1:  # Overbought condition
            reward -= 15  # Penalize for potential buy

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range