import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Extract closing prices (every 6th element starting from index 0)
    opening_prices = s[1::6]  # Extract opening prices (every 6th element starting from index 1)
    
    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - opening_prices[-1]) / opening_prices[-1] * 100 if opening_prices[-1] != 0 else 0
    
    # Feature 2: Short-term Moving Average (5-day)
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    
    # Feature 3: Relative Strength Index (RSI)
    gains = np.maximum(closing_prices[1:] - closing_prices[:-1], 0)
    losses = np.maximum(closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Combine features into a single array
    features = [price_change_pct, moving_average, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # MILD POSITIVE reward for SELL-aligned features
        return reward
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward