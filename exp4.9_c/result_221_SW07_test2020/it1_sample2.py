import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) over the last 20 days
    alpha = 2 / (20 + 1)
    ema = np.zeros(len(closing_prices))
    ema[0] = closing_prices[0]  # Start with the first closing price
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] - ema[i-1]) * alpha + ema[i-1]

    # Feature 2: Average True Range (ATR) over the last 14 days
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0

    # Feature 3: Z-score of Returns (standardized return)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    z_score = (np.mean(daily_returns) - np.mean(daily_returns[-20:])) / np.std(daily_returns[-20:]) if np.std(daily_returns[-20:]) != 0 else 0

    features = [ema[-1], atr, z_score]  # Return the latest EMA, ATR, and Z-score
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Define historical measures for dynamic thresholds (example)
    historical_std = np.std(features)  # Use the std of features as a measure
    high_risk_threshold = 0.7 * historical_std  # Relative threshold based on historical std
    low_risk_threshold = 0.4 * historical_std  # Another relative threshold
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # STRONG NEGATIVE reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive feature indicates a BUY signal
            reward -= np.random.uniform(30, 50)
        # MILD POSITIVE reward for SELL-aligned features
        if features[0] < 0:  # Assuming negative feature indicates a SELL signal
            reward += np.random.uniform(5, 10)

    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= np.random.uniform(10, 20)
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Positive momentum
                reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Negative momentum
                reward += np.random.uniform(10, 20)
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Assuming Z-score indicates oversold
            reward += np.random.uniform(10, 20)
        elif features[2] > 1:  # Assuming Z-score indicates overbought
            reward -= np.random.uniform(10, 20)
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds