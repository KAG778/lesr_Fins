import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Price Momentum (last 5 days)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0
    
    # Feature 2: Volume Change (last 5 days)
    volume_change = volumes[-1] - volumes[-6] if len(volumes) >= 6 else 0
    
    # Feature 3: Relative Strength Index (RSI)
    # Calculate average gain and loss over the last 14 days (simplified to 5 days here)
    gains = []
    losses = []
    
    for i in range(1, min(len(closing_prices), 6)):
        change = closing_prices[i] - closing_prices[i - 1]
        if change > 0:
            gains.append(change)
        else:
            losses.append(-change)
    
    avg_gain = np.mean(gains) if gains else 0
    avg_loss = np.mean(losses) if losses else 0
    
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs)) if avg_loss > 0 else 100  # RSI formula
    
    features = [price_momentum, volume_change, rsi]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY-aligned features
        reward += np.random.uniform(5, 10)   # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)    # moderate negative for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # positive reward for upward features
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)     # Reward mean-reversion features
        reward -= np.random.uniform(5, 15)     # Penalize breakout-chasing features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]