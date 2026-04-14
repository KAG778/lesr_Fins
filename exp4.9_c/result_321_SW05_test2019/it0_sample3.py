import numpy as np

def revise_state(s):
    # s: 120d raw state, where each day has 6 features: [close, open, high, low, volume, adj_close]
    num_days = 20
    closing_prices = s[0:num_days*6:6]  # Extract closing prices
    volumes = s[4:num_days*6:6]          # Extract volumes

    # Calculate RSI
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0).mean()
        loss = -np.where(delta < 0, delta, 0).mean()  # Use negative losses
        rs = gain / loss if loss > 0 else 0  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices[-14:])  # Using the last 14 days for RSI
    
    # Calculate SMA
    sma = np.mean(closing_prices[-10:])  # Simple moving average of the last 10 days

    # Calculate volume change (current volume vs average volume of last 5 days)
    avg_volume = np.mean(volumes[-5:]) if np.mean(volumes[-5:]) > 0 else 1  # Avoid division by zero
    recent_volume = volumes[-1]
    volume_change = (recent_volume - avg_volume) / avg_volume  # Relative change in volume

    # Prepare features array
    features = [rsi, sma, volume_change]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward for mean-reversion features (e.g., oversold → buy)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Constrain reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward