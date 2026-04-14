import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Closing prices are at indices 0, 6, 12, ...
    volumes = s[4::6]        # Volume at indices 4, 10, 16, ...

    # Feature 1: Price Momentum (Rate of Change)
    price_momentum = (closing_prices[-1] - closing_prices[0]) / closing_prices[0] if closing_prices[0] != 0 else 0

    # Feature 2: Volatility (Standard Deviation of Returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if len(returns) > 0 else 0

    # Feature 3: Volume Change (Percentage change from first to last day)
    volume_change = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] != 0 else 0

    features = [price_momentum, volatility, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        return -40  # STRONG NEGATIVE reward for risky BUY-aligned features
    elif risk_level > 0.4:
        return -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward = 10  # Positive reward for upward features
        else:
            reward = 10  # Positive reward for downward features
        return reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Here, we would reward mean-reversion features
        return 5  # Reward for mean-reversion logic

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        return 5 * 0.5  # Reduce reward magnitude by 50%

    return 0  # Default reward if no conditions are met