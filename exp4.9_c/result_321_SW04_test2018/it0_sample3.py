import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days OHLCV)
    
    # Closing prices
    closing_prices = s[0:120:6]  # Extract closing prices
    
    # Feature 1: Price Momentum (last close / close 5 days ago - 1) * 100
    if closing_prices[5] != 0:  # Avoid division by zero
        price_momentum = (closing_prices[0] / closing_prices[5] - 1) * 100
    else:
        price_momentum = 0  # or handle as desired
    
    # Feature 2: Average True Range (ATR)
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    
    # Feature 3: Relative Strength Index (RSI)
    gains = np.maximum(closing_prices[1:] - closing_prices[:-1], 0)
    losses = np.maximum(closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    if avg_loss == 0:
        rsi = 100  # If no losses, RSI is 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    features = [price_momentum, atr, rsi]
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
        if enhanced_s[123] > 0:  # BUY-aligned features
            return np.random.uniform(-50, -30)  # STRONG NEGATIVE reward
        else:  # SELL-aligned features
            return np.random.uniform(5, 10)  # MILD POSITIVE reward
    elif risk_level > 0.4:
        return np.random.uniform(-20, -10)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] > 0:  # Upward features
            reward += 10  # Positive reward
        elif trend_direction < -0.3 and enhanced_s[123] < 0:  # Downward features
            reward += 10  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] > 0:  # Oversold → BUY
            reward += 10
        else:  # Overbought → SELL
            reward += 10

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward