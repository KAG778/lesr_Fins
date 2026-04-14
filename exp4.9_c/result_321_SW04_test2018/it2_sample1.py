import numpy as np

def revise_state(s):
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
    
    # Feature 3: Z-score of the 14-day RSI to adapt to different regimes
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    rsi_z_score = (rsi - 50) / (np.std(rsi) if np.std(rsi) != 0 else 1)  # Z-score calculation
    
    # Feature 4: Volume Change Percentage
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 if len(volumes) > 1 and volumes[-2] != 0 else 0

    features = [price_momentum, atr, rsi_z_score, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive momentum indicates a BUY signal
            reward = np.random.uniform(-100, -50)  # Strong negative reward
        else:
            reward = np.random.uniform(5, 15)  # Mild positive reward for SELL signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Upward features & uptrend
            reward += 20 * (features[0] / historical_std)  # Positive reward for alignment
        elif trend_direction < -0.3 and features[0] < 0:  # Downward features & downtrend
            reward += 20 * (features[0] / historical_std)  # Positive reward for alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Assuming Z-score of RSI < -1 indicates oversold
            reward += 10  # Reward for buying in oversold conditions
        elif features[2] > 1:  # Assuming Z-score of RSI > 1 indicates overbought
            reward += 10  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward