import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    n = len(closing_prices)
    
    # Feature 1: Daily Returns (percentage change)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if n > 1 else np.array([0])
    daily_returns = np.concatenate(([0], daily_returns))  # Fill first element with 0 for shape compatibility
    
    # Feature 2: Average True Range (ATR) for volatility
    true_ranges = np.maximum(closing_prices[1:] - closing_prices[:-1], 
                              np.maximum(closing_prices[1:] - np.roll(closing_prices, 1)[1:], 
                                         np.roll(closing_prices, -1)[:-1] - closing_prices[:-1]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.mean(true_ranges) if len(true_ranges) > 0 else 0
    
    # Feature 3: 14-day RSI
    if len(closing_prices) < 14:
        rsi = np.zeros_like(closing_prices)
    else:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain) if len(gain) > 0 else 0
        avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss) if len(loss) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    
    features = [np.mean(daily_returns[-5:]), atr, rsi[-1]]  # Use latest RSI value
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0
    
    # Calculate dynamic thresholds based on historical volatility
    historical_volatility = np.std(features)  # Assuming features contain returns or relevant metrics
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_medium = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if features[0] > 0:  # Assuming first feature is aligned with BUY
            reward = np.random.uniform(-50, -30)  # Strong negative for risky BUY
        else:
            reward = np.random.uniform(5, 10)  # Mild positive for SELL
    elif risk_level > risk_threshold_medium:
        if features[0] > 0:  # BUY signal
            reward = np.random.uniform(-20, -10)  # Moderate negative for risky BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend & positive feature
            reward += 15  # Positive reward for correct trend-following
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend & negative feature
            reward += 15  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Assuming first feature is a SELL signal
            reward += 10  # Reward for mean-reversion selling
        elif features[0] > 0:  # Assuming first feature is a BUY signal
            reward -= 10  # Penalize for chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range