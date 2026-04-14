import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Momentum (Current close - Close 5 days ago)
    price_momentum = s[114] - s[108] if len(s) > 114 else 0
    features.append(price_momentum)

    # Feature 2: Average True Range (ATR) for volatility measure
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    close_prices = s[0:120:6]
    
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - close_prices[:-1]), 
                                        np.abs(low_prices[1:] - close_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) > 0 else 0
    features.append(atr)

    # Feature 3: RSI (Relative Strength Index)
    gains = []
    losses = []
    for i in range(1, len(close_prices)):
        change = close_prices[i] - close_prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            losses.append(-change)
            gains.append(0)

    avg_gain = np.mean(gains[-14:]) if len(gains) > 0 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0
    
    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY
        reward += 5.0 * (100 - features[2])  # Mild positive for SELL if RSI is low
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Price momentum is positive
            reward += 10.0 * features[0]  # Strong positive for aligned momentum
        elif features[0] < 0:  # Price momentum is negative
            reward += 10.0 * features[0]  # Penalize if momentum is against trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Oversold condition
            reward += 15.0  # Strong buy signal
        elif features[2] > 70:  # Overbought condition
            reward -= 10.0  # Moderate sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))