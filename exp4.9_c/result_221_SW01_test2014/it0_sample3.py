import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (every 6th element starting from index 0)
    
    # Calculate price change percentage
    price_change_percentage = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if closing_prices[i-1] != 0:  # Prevent division by zero
            price_change_percentage[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Calculate MACD
    short_ema = np.convolve(closing_prices, np.ones(12)/12, 'valid')  # 12-day EMA
    long_ema = np.convolve(closing_prices, np.ones(26)/26, 'valid')  # 26-day EMA
    macd = short_ema[-len(long_ema):] - long_ema  # Align lengths
    macd_signal = np.convolve(macd, np.ones(9)/9, 'valid')  # Signal line (9-day EMA of MACD)
    
    # Calculate RSI
    gains = np.where(price_change_percentage > 0, price_change_percentage, 0)
    losses = np.where(price_change_percentage < 0, -price_change_percentage, 0)
    
    avg_gain = np.mean(gains[-14:])  # 14-day average gain
    avg_loss = np.mean(losses[-14:])  # 14-day average loss
    rs = avg_gain / avg_loss if avg_loss != 0 else 0  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Compile features
    features = [
        price_change_percentage[-1],  # Most recent price change percentage
        macd[-1] if len(macd) > 0 else 0,  # Most recent MACD value
        rsi  # Most recent RSI value
    ]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward = -40  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward = -10  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)