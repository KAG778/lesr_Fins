import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extract every 6th element starting from index 0 (closing prices)
    volumes = s[4::6]          # Extract every 6th element starting from index 4 (volumes)
    
    # Feature 1: 10-day Momentum
    momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0
    
    # Feature 2: Exponential Moving Average (EMA) of closing prices (last 10 days)
    weights = np.exp(np.linspace(-1., 0., 10))
    weights /= weights.sum()
    ema = np.dot(weights, closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    
    # Feature 3: Bollinger Bands (Upper and Lower) - 20-day period
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_avg + (2 * std_dev)
        lower_band = moving_avg - (2 * std_dev)
    else:
        upper_band, lower_band = np.nan, np.nan
    
    # Feature 4: Volume Change (current vs. average volume over the last 10 days)
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    
    # Compile features into a single array
    features = [momentum, ema, upper_band, lower_band, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculating historical thresholds using a standard deviation approach
    mean_risk = 0.5  # Assuming we have a historical mean for risk level
    std_risk = 0.2   # Assuming we have a historical std for risk level
    risk_threshold = mean_risk + 1 * std_risk  # Example threshold based on std deviation
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)  # MILD POSITIVE reward for SELL-aligned features
        return np.clip(reward, -100, 100)
    elif risk_level > mean_risk:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < mean_risk:
        if trend_direction > 0:
            reward += np.random.uniform(10, 25)  # Positive reward for bullish momentum
        elif trend_direction < 0:
            reward += np.random.uniform(10, 25)  # Positive reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Positive reward for mean-reversion features
        reward -= np.random.uniform(5, 15)  # Negative for chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < mean_risk:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]