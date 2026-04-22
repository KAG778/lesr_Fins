import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    
    # Feature 1: Price Change (%) from previous day (day i to day i-1)
    price_change_percent = np.zeros(19)  # There are 19 changes (from day 0 to day 1, ..., day 18 to day 19)
    for i in range(1, len(closing_prices)):
        if closing_prices[i-1] != 0:  # Prevent division by zero
            price_change_percent[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1] * 100
    
    # Feature 2: 10-day Moving Average (last 10 days)
    moving_average = np.mean(closing_prices[-10:])  # Average of last 10 days closing prices
    
    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)  # Changes in price
    gain = np.where(delta > 0, delta, 0).mean()  # Average gain
    loss = np.where(delta < 0, -delta, 0).mean()  # Average loss
    rs = gain / loss if loss != 0 else 0  # Relative strength
    rsi = 100 - (100 / (1 + rs))  # Calculate RSI

    # Combine features into a single array
    features = [
        np.mean(price_change_percent),  # Average price change percentage
        moving_average,
        rsi
    ]

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
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)   # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Reward for upward trend
        elif trend_direction < -0.3:
            reward += 10  # Reward for downward trend
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward for mean reversion features
        reward -= 5  # Penalize breakout-chasing features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]