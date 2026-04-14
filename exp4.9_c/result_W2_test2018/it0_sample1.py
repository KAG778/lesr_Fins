import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Price Momentum (current close - close 5 days ago) / close 5 days ago
    # Ensure not to divide by zero
    if len(closing_prices) > 5 and closing_prices[5] != 0:
        price_momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5]
    else:
        price_momentum = 0.0
    
    # Feature 2: Volatility Indicator (standard deviation of closing prices)
    volatility_indicator = np.std(closing_prices)

    # Feature 3: Volume Change (current volume / mean volume over last 5 days)
    if len(volumes) >= 5:
        avg_volume = np.mean(volumes[1:6])  # Average of last 5 days
        if avg_volume != 0:
            volume_change = volumes[0] / avg_volume
        else:
            volume_change = 1.0  # Default to 1 if avg volume is zero
    else:
        volume_change = 1.0  # Default to 1 if not enough data

    features = [price_momentum, volatility_indicator, volume_change]
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
        reward = np.random.uniform(-50, -30)  # STRONG NEGATIVE reward for BUY
    elif risk_level > 0.4:
        reward = np.random.uniform(-10, -5)   # moderate negative reward for BUY
    
    # If risk is low, check the next priorities
    if risk_level <= 0.4:
        if abs(trend_direction) > 0.3:
            # Priority 2 — TREND FOLLOWING
            if trend_direction > 0.3:
                # Reward upward features (assume features[0] is the momentum)
                reward += 10 if enhanced_s[123] > 0 else 0  # Example condition for upward feature
            else:
                # Reward downward features
                reward += 10 if enhanced_s[123] < 0 else 0  # Example condition for downward feature
        elif abs(trend_direction) < 0.3:
            # Priority 3 — SIDEWAYS / MEAN REVERSION
            # Reward mean-reversion features (assume features[0] is the momentum)
            reward += 5 if enhanced_s[123] < 0 else -5  # Example for mean-reversion
            
        # Priority 4 — HIGH VOLATILITY
        if volatility_level > 0.6:
            reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return float(reward)