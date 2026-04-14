import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices (every 6th element)
    n = len(closing_prices)
    
    # Feature 1: Exponential Moving Average (EMA) - 10 periods
    ema = np.mean(closing_prices[-10:]) if n >= 10 else closing_prices[-1]
    
    # Feature 2: Bollinger Bands Width - 20 periods
    if n >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        bollinger_band_width = (upper_band - lower_band) / moving_average
    else:
        bollinger_band_width = 0

    # Feature 3: Rate of Change (ROC) - 5 periods
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100 if n >= 6 and closing_prices[-6] != 0 else 0
    
    # Feature 4: Volume Change Percentage
    volumes = s[4:120:6]  # Extract volumes
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 if len(volumes) > 1 and volumes[-2] != 0 else 0
    
    features = [ema, bollinger_band_width, roc, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate historical std for relative thresholds
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    z_score_roc = (features[2] - np.mean(features[2])) / historical_std if historical_std > 0 else 0  # Z-score for ROC

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if z_score_roc > 0:  # Positive momentum indicates a BUY signal
            reward = np.random.uniform(-100, -50)  # Strong negative reward
        else:  # Negative momentum indicates a SELL signal
            reward = np.random.uniform(5, 15)  # Mild positive reward

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and z_score_roc > 0:  # Aligning with bullish momentum
            reward += 20 * (features[2] / historical_std)  # Reward for alignment
        elif trend_direction < -0.3 and z_score_roc < 0:  # Aligning with bearish momentum
            reward += 20 * (features[2] / historical_std)  # Reward for alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if z_score_roc < -1:  # Oversold condition
            reward += 10  # Reward for buying in oversold conditions
        elif z_score_roc > 1:  # Overbought condition
            reward += 10  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return max(min(reward, 100), -100)