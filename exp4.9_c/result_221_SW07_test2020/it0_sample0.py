import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extracting closing prices
    volumes = s[4:120:6]          # Extracting volumes
    
    # Feature 1: Price Change Ratio (current closing to previous closing)
    price_change_ratio = closing_prices[0] / closing_prices[1] if closing_prices[1] != 0 else 0
    
    # Feature 2: Average Trading Volume over the last 20 days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0
    
    # Feature 3: Relative Strength Index (RSI)
    gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
    losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
    
    average_gain = np.mean(gains[-14:]) if len(gains) > 0 else 0
    average_loss = np.mean(losses[-14:]) if len(losses) > 0 else 0
    
    # Avoid division by zero
    rs = average_gain / average_loss if average_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Return the computed features
    features = [price_change_ratio, average_volume, rsi]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += np.random.uniform(-50, -30)  # Strong negative reward for BUY signals
        # Assuming some condition for SELL-aligned features
        reward += np.random.uniform(5, 10)  # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += np.random.uniform(-15, -5)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features (oversold → buy, overbought → sell)
        reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)