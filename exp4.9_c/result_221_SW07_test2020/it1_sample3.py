import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Price Momentum (Rate of Change over the last 19 days)
    momentum = (closing_prices[0] - closing_prices[19]) / closing_prices[19] if closing_prices[19] != 0 else 0
    
    # Feature 2: Average Volume Change over the last 20 days
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    last_volume = volumes[0]
    volume_change = (last_volume - average_volume) / average_volume if average_volume != 0 else 0
    
    # Feature 3: Average True Range (ATR) as a volatility measure
    high_prices = s[2::6]
    low_prices = s[3::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], high_prices[1:] - closing_prices[:-1])
    true_ranges = np.maximum(true_ranges, closing_prices[:-1] - low_prices[1:])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    
    features = [momentum, volume_change, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate historical standard deviations for relative thresholds
    historical_std = np.std(features)
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive momentum
            reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        if features[0] < 0:  # Negative momentum
            reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        if features[0] > 0:
            reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and positive momentum
            reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and negative momentum
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += np.random.uniform(10, 20)
        elif features[0] > 0:  # Overbought condition
            reward -= np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds