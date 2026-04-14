import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volume
    
    # Feature 1: Price Momentum (last closing price - closing price 5 days ago)
    momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) > 5 else 0
    
    # Feature 2: Volume Change (percentage change in volume over the last 5 days)
    recent_volume = volumes[0]
    avg_volume = np.mean(volumes[:5]) if len(volumes) > 5 else 1  # Avoid division by zero
    volume_change = (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
    
    # Feature 3: Bollinger Band Width (using simple moving average and standard deviation)
    sma = np.mean(closing_prices[:5]) if len(closing_prices) > 5 else 0
    std_dev = np.std(closing_prices[:5]) if len(closing_prices) > 5 else 0
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    bollinger_band_width = upper_band - lower_band if upper_band and lower_band else 0

    features = [momentum, volume_change, bollinger_band_width]
    
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
            reward = np.random.uniform(-50, -30)
        elif enhanced_s[123] < 0:  # SELL-aligned features
            reward = np.random.uniform(5, 10)
        return reward
    
    if risk_level > 0.4:
        if enhanced_s[123] > 0:  # BUY signals
            reward = np.random.uniform(-20, -10)
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 30)  # Upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 30)  # Downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Overbought → sell
            reward += np.random.uniform(5, 15)
        elif enhanced_s[123] > 0:  # Oversold → buy
            reward += np.random.uniform(5, 15)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds