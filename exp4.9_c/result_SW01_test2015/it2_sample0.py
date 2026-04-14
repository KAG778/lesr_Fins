import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Exponential Moving Average (EMA) for trend detection
    ema = np.zeros_like(closing_prices)
    alpha = 2 / (5 + 1)  # Smoothing for a 5-day EMA
    ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    features.append(ema[-1])

    # Feature 2: Average True Range (ATR) for volatility measurement
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0.0
    features.append(atr)

    # Feature 3: Z-score of recent returns to assess momentum
    recent_returns = np.diff(closing_prices) / closing_prices[:-1]
    z_score = (recent_returns[-1] - np.mean(recent_returns)) / np.std(recent_returns) if np.std(recent_returns) != 0 else 0
    features.append(z_score)

    # Feature 4: Moving Average Convergence Divergence (MACD) for trend confirmation
    short_ema = np.zeros_like(closing_prices)
    long_ema = np.zeros_like(closing_prices)
    short_alpha = 2 / (12 + 1)  # Short EMA for MACD
    long_alpha = 2 / (26 + 1)    # Long EMA for MACD
    short_ema[0] = closing_prices[0]
    long_ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        short_ema[i] = (closing_prices[i] * short_alpha) + (short_ema[i - 1] * (1 - short_alpha))
        long_ema[i] = (closing_prices[i] * long_alpha) + (long_ema[i - 1] * (1 - long_alpha))
    macd = short_ema - long_ema
    features.append(macd[-1])

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Your computed features from revise_state
    reward = 0.0

    # Calculate dynamic thresholds based on historical features
    historical_std = np.std(features)
    historical_mean = np.mean(features)
    
    risk_threshold_high = historical_mean + 0.7 * historical_std
    risk_threshold_low = historical_mean + 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for high risk
        reward += 10 if features[2] < 0 else 0  # Mild positive for selling when momentum is against buying
        return float(np.clip(reward, -100, 100))  # Early exit
    elif risk_level > risk_threshold_low:
        reward -= np.random.uniform(5, 15)  # Moderate penalty for buying

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_low:
        reward += 10 * features[2]  # Align with momentum direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10 if features[2] < 0 else -10  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits