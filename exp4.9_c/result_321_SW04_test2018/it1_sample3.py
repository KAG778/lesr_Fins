import numpy as np

def revise_state(s):
    # s: 120d raw state from which we will extract features
    closing_prices = s[0:120:6]  # Extract closing prices (day i at index 6*i)
    opening_prices = s[1:120:6]  # Extract opening prices (day i at index 6*i + 1)
    volumes = s[4:120:6]          # Extract volumes (day i at index 6*i + 4)

    # Feature 1: Price Change Momentum (most recent close vs 5 days ago)
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100 if len(closing_prices) > 5 else 0

    # Feature 2: Average True Range (ATR) for volatility
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
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

    features = [momentum, atr, rsi_z_score, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate historical standard deviation for relative thresholds
    if len(features) > 0:
        historical_std = np.std(features)
    else:
        historical_std = 1  # To avoid division by zero

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive price change indicates a BUY signal
            reward = np.random.uniform(-100, -50)  # Strong negative reward
        # Mild positive reward for SELL-aligned features
        else:
            reward = np.random.uniform(5, 15)  # Mild positive reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward = np.random.uniform(-30, -10)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Upward features & uptrend
            reward += 20 * (features[0] / historical_std)  # Reward aligned with momentum
        elif trend_direction < -0.3 and features[0] < 0:  # Downward features & downtrend
            reward += 20 * (features[0] / historical_std)  # Reward aligned with momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Assuming oversold condition for Z-score of RSI
            reward += 10  # Reward for buying in oversold conditions
        elif features[2] > 1:  # Assuming overbought condition for Z-score of RSI
            reward += 10  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward