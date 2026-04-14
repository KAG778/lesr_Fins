import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    opening_prices = s[1::6]  # Extract opening prices (every 6th element starting from index 1)
    volumes = s[4::6]         # Extract volumes (every 6th element starting from index 4)
    
    # Feature 1: Price Momentum (closing - opening)
    price_momentum = closing_prices - opening_prices
    
    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(volumes)
    
    # Feature 3: Volatility (standard deviation of closing prices)
    volatility = np.std(closing_prices)

    # Handle edge cases for features if necessary
    # For average volume, we can check if the volume array is not empty
    if np.isnan(avg_volume) or avg_volume == 0:
        avg_volume = 1e-6  # Avoid division by zero in future calculations
    if np.isnan(volatility):
        volatility = 0.0  # If there is no variation, set volatility to 0

    # Return only the new features as a 1D numpy array
    features = [price_momentum[-1], avg_volume, volatility]  # We use the most recent price momentum
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
        # STRONG NEGATIVE reward for BUY-aligned features
        reward += -40  # Example strong negative penalty for buying in high-risk
        # MILD POSITIVE reward for SELL-aligned features
        reward += 7  # Example mild positive for selling in high-risk
        return reward
    
    if risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -15  # Example moderate penalty for buying in elevated risk
        return reward
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward trend
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward trend (correct bearish bet)
        return reward
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Example conditions for mean-reversion rewards
        reward += 5  # Example positive for mean-reversion features
        return reward

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward