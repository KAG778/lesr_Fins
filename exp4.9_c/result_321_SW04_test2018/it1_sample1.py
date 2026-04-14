import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    
    # Feature 1: Price Momentum (last close / close 5 days ago - 1) * 100
    price_momentum = (closing_prices[-1] / closing_prices[-6] - 1) * 100 if len(closing_prices) > 5 else 0
    
    # Feature 2: Average True Range (ATR) - measure of volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    
    # Feature 3: Z-score of the RSI (Relative Strength Index)
    gains = np.maximum(closing_prices[1:] - closing_prices[:-1], 0)
    losses = np.maximum(closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    rsi_zscore = (rsi - 50) / np.std(rsi) if np.std(rsi) != 0 else 0

    features = [price_momentum, atr, rsi_zscore]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive price momentum indicates a BUY signal
            reward = np.random.uniform(-50, -30)  # Strong negative reward
        else:
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL-aligned features
            
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Upward features & uptrend
            reward += np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < -0.3 and features[0] < 0:  # Downward features & downtrend
            reward += np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Assume Z-score of RSI < -1 indicates oversold
            reward += 10  # Reward for buying in oversold conditions
        elif features[2] > 1:  # Assume Z-score of RSI > 1 indicates overbought
            reward += 10  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return float(reward)