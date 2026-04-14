import numpy as np

def revise_state(s):
    # s: 120d raw state
    # We will compute features based on the last 20 days of closing prices
    closing_prices = s[0::6]  # Extract closing prices
    
    # Feature 1: Price Change Percentage
    price_change_percentage = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if closing_prices[i-1] != 0:
            price_change_percentage[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Feature 2: 5-day Moving Average
    moving_average = np.zeros(20)
    for i in range(4, 20):
        moving_average[i] = np.mean(closing_prices[i-4:i+1])
    
    # Feature 3: Relative Strength Index (RSI)
    rsi = np.zeros(20)
    gains = np.zeros(20)
    losses = np.zeros(20)
    
    for i in range(1, 20):
        change = closing_prices[i] - closing_prices[i-1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = -change
    
    average_gain = np.mean(gains[-14:]) if np.sum(gains[-14:]) > 0 else 0
    average_loss = np.mean(losses[-14:]) if np.sum(losses[-14:]) > 0 else 0
    
    if average_loss == 0:
        rsi[14:] = 100  # RSI is 100 if no losses
    else:
        rs = average_gain / average_loss
        rsi[14:] = 100 - (100 / (1 + rs))
    
    # Return only the computed features
    features = np.concatenate((price_change_percentage, moving_average, rsi))
    return features

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
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)   # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for bullish features
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the range