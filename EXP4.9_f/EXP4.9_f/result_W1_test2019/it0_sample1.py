import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price Momentum (latest closing - closing price 5 days ago)
    price_momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) > 5 else 0

    # Feature 2: Volatility (Standard Deviation of the last 20 closing prices)
    volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0

    # Feature 3: Volume Change (Current volume - Previous volume)
    volume_change = volumes[0] - volumes[1] if len(volumes) > 1 else 0
    
    # Return the features as a numpy array
    return np.array([price_momentum, volatility, volume_change])

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
        reward += -30  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward += -10  # Moderate negative for BUY signals

    # If risk level is acceptable, apply further reward logic
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:
                reward += 10  # Positive reward for upward trend
            else:
                reward += 10  # Positive reward for downward trend

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Implement mean-reversion logic (pseudo logic for oversold/overbought)
            # Here, we could use features to assess overbought/oversold conditions
            reward += 5  # Example positive reward for mean-reverting features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Limit the reward within the range [-100, 100]
    return np.clip(reward, -100, 100)