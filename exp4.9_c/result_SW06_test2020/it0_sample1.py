import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (every 6th element)
    
    # Feature 1: Price Change (percentage change from previous closing price)
    price_change = np.zeros(19)  # We can only compute this for 19 days
    for i in range(1, 20):
        price_change[i - 1] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1] if closing_prices[i - 1] != 0 else 0
    
    # Feature 2: 5-Day Moving Average
    moving_average = np.zeros(20)
    for i in range(4, 20):
        moving_average[i] = np.mean(closing_prices[i - 4:i + 1])
    
    # Feature 3: Relative Strength Index (RSI)
    gains = np.zeros(19)
    losses = np.zeros(19)
    for i in range(1, 20):
        change = closing_prices[i] - closing_prices[i - 1]
        if change > 0:
            gains[i - 1] = change
        else:
            losses[i - 1] = -change
    
    average_gain = np.mean(gains) if np.mean(gains) != 0 else 0
    average_loss = np.mean(losses) if np.mean(losses) != 0 else 0
    
    rs = average_gain / average_loss if average_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Combine features
    features = []
    features.extend(price_change)
    features.extend(moving_average)
    features.append(rsi)  # Add RSI as the last feature
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward = -np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY
    elif risk_level > 0.4:
        reward = -np.random.uniform(5, 15)  # MODERATE NEGATIVE for BUY
    
    # If risk level is low, check for trend following and mean reversion
    if risk_level < 0.4:
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                reward += 10  # Positive reward for upward trend
            elif trend_direction < -0.3:
                reward += 10  # Positive reward for downward trend
        elif abs(trend_direction) < 0.3:
            # Check for mean reversion
            # Assuming features contain signals for mean-reversion strategies
            reward += 5  # Reward for mean-reversion features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)