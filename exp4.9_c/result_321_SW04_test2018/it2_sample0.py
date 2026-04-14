import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    volumes = s[4:120:6]          # Extract volumes

    # Feature 1: Rate of Change (ROC) of closing prices
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 5 and closing_prices[-6] != 0 else 0

    # Feature 2: Exponential Moving Average (EMA)
    ema_period = 10
    ema = np.mean(closing_prices[-ema_period:]) if len(closing_prices) >= ema_period else closing_prices[-1]

    # Feature 3: Average True Range (ATR)
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0

    # Feature 4: Z-Score of recent price changes
    price_changes = np.diff(closing_prices)
    z_score = (price_changes[-1] - np.mean(price_changes[-20:])) / (np.std(price_changes[-20:]) if len(price_changes) >= 20 else 1e-10)

    # Feature 5: Volume Change Percentage
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 if len(volumes) > 1 and volumes[-2] != 0 else 0

    features = [roc, ema, atr, z_score, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]  # Extract regime information
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(features) if len(features) > 0 else 1
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive ROC indicates a BUY signal
            reward = np.random.uniform(-100, -50)  # Strong negative reward
        else:
            reward = np.random.uniform(10, 20)  # Mild positive reward for SELL signals
    elif risk_level > 0.4:
        if features[0] > 0:  # Positive ROC indicates a BUY signal
            reward = np.random.uniform(-30, -10)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish alignment
            reward += 20 * (features[0] / historical_std)  # Reward aligned with momentum
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish alignment
            reward += 20 * (features[0] / historical_std)  # Reward aligned with momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < -1:  # Oversold condition (Z-score of recent changes)
            reward += 10  # Reward for buying in oversold conditions
        elif features[3] > 1:  # Overbought condition (Z-score of recent changes)
            reward += 10  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward