import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    
    # Feature 1: Average True Range (ATR)
    high_prices = s[1::6]  # Extract high prices
    low_prices = s[2::6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR
    
    # Feature 2: Bollinger Bands (Upper Band)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices[-20:]) > 0 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices[-20:]) > 1 else 0
    upper_band = moving_average + (2 * std_dev)  # 2 standard deviations from the mean

    # Feature 3: MACD (Moving Average Convergence Divergence)
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices[-12:]) > 0 else 0  # 12-day EMA
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices[-26:]) > 0 else 0  # 26-day EMA
    macd = short_ema - long_ema

    features = [atr, upper_band, macd]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)  # Mild negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Positive trend
            reward += np.random.uniform(10, 20)  # Positive for upward momentum
        else:  # Negative trend
            reward += np.random.uniform(10, 20)  # Positive for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += np.random.uniform(5, 15)  # Reward mean-reversion logic
        reward -= np.random.uniform(5, 15)  # Penalize for breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility
    
    # Ensure reward stays within bounds of [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward