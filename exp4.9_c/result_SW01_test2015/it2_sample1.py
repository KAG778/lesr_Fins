import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Exponential Moving Average (EMA) over 10 days to capture trends
    ema = np.zeros(20)
    alpha = 2 / (10 + 1)  # Smoothing factor for 10-day EMA
    ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    features.append(ema[-1])  # Most recent EMA

    # Feature 2: Average True Range (ATR) to measure volatility
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-5:]) if len(true_ranges) > 0 else 0.0  # ATR over last 5 days
    features.append(atr)

    # Feature 3: Z-score of recent returns to measure momentum
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    z_score = (daily_returns[-1] - np.mean(daily_returns)) / (np.std(daily_returns) if np.std(daily_returns) != 0 else 1)
    features.append(z_score)

    # Feature 4: Relative Strength Index (RSI) to assess overbought/oversold conditions
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()
    loss = -np.where(delta < 0, delta, 0).mean()
    rs = gain / loss if loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Your computed features from revise_state
    reward = 0.0

    # Calculate dynamic thresholds based on historical features
    historical_volatility = np.std(features)  # Historical volatility from features
    high_risk_threshold = 0.7 * historical_volatility
    medium_risk_threshold = 0.4 * historical_volatility
    trend_threshold = 0.3 * np.std(features[:3])  # Using first three features for trend threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for high risk
        reward += 10 if features[0] < 0 else 0  # Positive for selling if price is falling
        return float(np.clip(reward, -100, 100))  # Early exit
    elif risk_level > medium_risk_threshold:
        reward -= np.random.uniform(5, 15)  # Moderate penalty for medium risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < medium_risk_threshold:
        reward += 20 * np.sign(trend_direction)  # Reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15 if features[3] < 30 else -15  # Reward mean-reversion based on RSI

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits