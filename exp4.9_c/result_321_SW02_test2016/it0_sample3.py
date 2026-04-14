import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # closing prices for 20 days
    volumes = s[4:120:6]          # trading volumes for 20 days
    
    # Feature 1: Price Momentum (5-day rate of change)
    price_momentum = np.zeros(19)  # 19 changes for 20 days
    for i in range(1, 20):
        if closing_prices[i-1] != 0:
            price_momentum[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
        else:
            price_momentum[i-1] = 0  # handle division by zero

    # Feature 2: Average Volume over the last 5 days
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0

    # Feature 3: Price Relative Strength Index (RSI)
    gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
    losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
    
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Compile features into a single array
    features = [np.mean(price_momentum), avg_volume, rsi]
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
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        return reward
    elif risk_level > 0.4:
        reward += -10  # moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Here, we would check for mean-reversion features
        reward += 5  # reward for mean-reversion features (simplified)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward