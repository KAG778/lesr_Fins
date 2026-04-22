import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices for the last 20 days
    closing_prices = s[0::6]  # OHLCV state, closing prices every 6th element
    daily_returns = np.zeros(20)
    
    # Calculate daily returns
    for i in range(20):
        if s[i * 6 + 0] > 0:  # Avoid division by zero
            daily_returns[i] = (s[i * 6 + 0] - s[i * 6 + 1]) / s[i * 6 + 1]

    # Feature 1: Average daily return over the last 20 days
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)

    # Feature 2: Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns)

    # Feature 3: Price momentum (current close vs. close 5 days ago)
    if s[5 * 6 + 0] > 0:
        price_momentum = (s[0] - s[5 * 6 + 0]) / s[5 * 6 + 0]
    else:
        price_momentum = 0
    features.append(price_momentum)

    # Feature 4: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 5: Z-score of the last 20 days' closing prices
    mean_price = np.mean(closing_prices)
    z_score = (closing_prices[-1] - mean_price) / (np.std(closing_prices) + 1e-10)
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Calculate relative thresholds based on historical volatility
    historical_volatility = np.std(features[0:20])  # Assuming features[0:20] contains daily returns
    volatility_threshold = historical_volatility * 1.5  # Example multiplier to adjust sensitivity

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # If feature[0] is aligned with BUY
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > 0.4:
        if features[0] > 0:  # If feature[0] is aligned with BUY
            reward = -20  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Align with upward trend
            reward += 10  # Positive reward for upward features
        elif trend_direction < 0 and features[0] < 0:  # Align with downward trend
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # RSI indicating oversold condition
            reward += 10  # Positive reward for mean reversion buy
        elif features[2] > 70:  # RSI indicating overbought condition
            reward -= 10  # Penalize for breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds