import numpy as np

def revise_state(s):
    # Extract closing prices (s[i*6 + 0])
    closing_prices = s[::6]  
    
    # Feature 1: Exponential Moving Average (EMA) for trend detection (20-day)
    ema = np.zeros(len(closing_prices))
    alpha = 2 / (20 + 1)  # Smoothing factor for 20-day EMA
    ema[0] = closing_prices[0]
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))

    # Feature 2: Average True Range (ATR) for volatility measurement (14-day)
    tr = np.zeros(len(closing_prices) - 1)
    for i in range(1, len(closing_prices)):
        high = s[i * 6 + 2]  # high prices
        low = s[i * 6 + 3]   # low prices
        prev_close = closing_prices[i - 1]
        tr[i - 1] = max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    atr = np.zeros(len(closing_prices))
    for i in range(13, len(closing_prices)):
        atr[i] = np.mean(tr[i - 13:i + 1])

    # Feature 3: Z-score of recent returns for mean-reversion signal
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_return = np.mean(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    std_return = np.std(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    
    features = [ema[-1], atr[-1], z_score]  # Return only the latest values
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Define historical thresholds based on volatility level
    risk_threshold = 0.7
    trend_threshold = 0.3
    low_risk_threshold = 0.4

    # Initialize reward
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold:
        reward += -np.random.uniform(30, 50)  # Strong negative for BUY
        reward += np.random.uniform(10, 20)  # Mild positive for SELL
    elif risk_level > low_risk_threshold:
        reward += -np.random.uniform(5, 15)  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < low_risk_threshold:
        if trend_direction > trend_threshold:  # Strong uptrend
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:  # Strong downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion strategies
        reward -= 10  # Penalize for breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the range [-100, 100]
    return np.clip(reward, -100, 100)