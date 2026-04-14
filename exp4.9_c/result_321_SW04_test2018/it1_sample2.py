import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days OHLCV)
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    volumes = s[4:120:6]          # Extract volumes
    
    # Feature 1: Price Momentum (percentage change over the last 5 days)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100 if len(closing_prices) > 5 and closing_prices[-6] != 0 else 0
    
    # Feature 2: Average True Range (ATR) for volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    
    # Feature 3: Volume Change (percentage change from the previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if len(volumes) > 1 and volumes[-2] > 0 else 0

    # Feature 4: Enhanced RSI with historical standard deviation
    gains = np.maximum(closing_prices[1:] - closing_prices[:-1], 0)
    losses = np.maximum(closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 5: Z-score of RSI to adapt to different market conditions
    rsi_std = np.std(rsi) if len(gains) > 0 else 1  # Avoid division by zero
    rsi_zscore = (rsi - np.mean(gains)) / rsi_std if rsi_std != 0 else 0

    features = [price_momentum, atr, volume_change, rsi_zscore]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Calculate historical volatility for relative threshold
    historical_volatility = np.std(features[0]) if len(features) > 0 else 1
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive price momentum indicates a BUY signal
            reward = np.random.uniform(-100, -50)  # STRONG NEGATIVE reward
        else:
            reward = np.random.uniform(5, 15)  # MILD POSITIVE reward for SELL signals
            
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish alignment
            reward += np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish alignment
            reward += np.random.uniform(10, 20)  # Positive reward
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < -1:  # Oversold condition (Z-score of RSI)
            reward += 10  # Reward for buying in oversold conditions
        elif features[3] > 1:  # Overbought condition (Z-score of RSI)
            reward += 10  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)

    return reward