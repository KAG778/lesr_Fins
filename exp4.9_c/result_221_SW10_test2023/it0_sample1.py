import numpy as np

def revise_state(s):
    # Handling edge cases
    closing_prices = s[0:120:6]  # Extracting closing prices for the last 20 days
    volumes = s[4:120:6]  # Extracting trading volumes for the last 20 days
    
    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) > 5 else 0
    
    # Feature 2: Relative Strength Index (RSI) calculation
    gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
    losses = -np.where(np.diff(closing_prices) < 0, np.diff(closing_prices), 0)
    
    average_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    average_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    
    rs = average_gain / average_loss if average_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 3: Volume Change (% change from previous day)
    volume_change = (volumes[0] - volumes[1]) / volumes[1] * 100 if volumes[1] != 0 else 0
    
    return np.array([price_momentum, rsi, volume_change])

def intrinsic_reward(enhanced_state):
    # Read regime_vector
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward = -40  # Midpoint of the range -30 to -50
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -20  # Example value for moderate penalty

    # If risk is low, check other priorities
    if risk_level < 0.4:
        if abs(trend_direction) > 0.3:
            # Priority 2 — TREND FOLLOWING
            if trend_direction > 0.3:
                reward += 20  # Positive reward for upward trend
            elif trend_direction < -0.3:
                reward += 20  # Positive reward for downward trend
        elif abs(trend_direction) < 0.3:
            # Priority 3 — SIDEWAYS / MEAN REVERSION
            reward += 15  # Reward for mean-reversion features (oversold/buy, overbought/sell)
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)