import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate the daily returns
    daily_returns = np.zeros(20)
    for i in range(20):
        if s[i * 6 + 0] > 0:  # Avoid division by zero
            daily_returns[i] = (s[i * 6 + 0] - s[i * 6 + 1]) / s[i * 6 + 1]  # Close - Open / Open

    # Feature 1: Average daily return over the last 20 days
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)

    # Feature 2: Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    # Feature 3: Price momentum (current close vs. close 5 days ago)
    if s[5 * 6 + 0] > 0:  # Ensure that the price 5 days ago is not zero
        price_momentum = (s[0] - s[5 * 6 + 0]) / s[5 * 6 + 0]  # Current Close - Close 5 days ago / Close 5 days ago
    else:
        price_momentum = 0  # Handle edge case
    features.append(price_momentum)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if enhanced_s[123] < 0:  # BUY-aligned feature
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:  # SELL-aligned feature
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > 0.4:
        if enhanced_s[123] < 0:  # BUY-aligned feature
            reward = -10  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and enhanced_s[123] > 0:  # Uptrend and upward feature
            reward += 10  # Positive reward
        elif trend_direction < 0 and enhanced_s[123] < 0:  # Downtrend and downward feature
            reward += 10  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] > 0:  # Oversold condition
            reward += 10  # Positive reward for mean-reversion buy
        elif enhanced_s[123] < 0:  # Overbought condition
            reward += -10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]