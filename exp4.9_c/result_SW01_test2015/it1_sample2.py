import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) - gives more weight to recent prices
    ema = np.zeros(20)
    alpha = 2 / (5 + 1)  # Using a span of 5 days
    ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    features.append(ema[-1])  # Current EMA
    
    # Feature 2: Average True Range (ATR) for volatility measure
    high = s[1::6]  # High prices
    low = s[2::6]   # Low prices
    tr = np.maximum(high[1:] - low[1:], high[1:] - closing_prices[:-1], closing_prices[:-1] - low[1:])
    atr = np.mean(tr[-5:])  # ATR over the last 5 days
    features.append(atr)

    # Feature 3: Z-score of recent returns (standardized return)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    z_score = (daily_returns[-1] - np.mean(daily_returns)) / np.std(daily_returns)
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features from revised state
    reward = 0.0
    
    # Calculate dynamic thresholds based on historical features
    risk_threshold = 0.7 * np.std(features) + np.mean(features)  # Dynamic threshold for risk
    trend_threshold = 0.3 * np.std(features)  # Dynamic threshold for trend
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for BUY
        reward += 10 if features[0] < 0 else 0  # Mild positive for SELL
        return np.clip(reward, -100, 100)  # Early exit
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        reward += 20 if trend_direction > 0 else -20  # Positive reward for trend alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15 if features[2] < 0 else -15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude
    
    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits