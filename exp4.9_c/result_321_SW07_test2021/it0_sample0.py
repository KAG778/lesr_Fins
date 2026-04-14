import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (indices 0, 6, 12, ..., 114)
    volumes = s[4::6]          # Extract trading volumes (indices 4, 10, 16, ..., 114)
    
    # Feature 1: Price Change Ratio
    price_change_ratio = closing_prices[0] / closing_prices[1] if closing_prices[1] != 0 else 0

    # Feature 2: Average Volume
    average_volume = np.mean(volumes) if volumes.size > 0 else 0

    # Feature 3: Price Momentum (current closing price - closing price 5 days ago)
    price_momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if closing_prices[5] != 0 else 0

    # Return computed features as a numpy array
    features = [price_change_ratio, average_volume, price_momentum]
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
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        # Optionally, assess SELL-aligned features here if applicable
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (implement logic based on features)
        reward += 5  # Example positive reward for mean reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]