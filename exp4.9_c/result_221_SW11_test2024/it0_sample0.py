import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices are at indices 0, 6, 12, ...
    opening_prices = s[1::6]   # Opening prices are at indices 1, 7, 13, ...
    high_prices = s[2::6]      # High prices are at indices 2, 8, 14, ...
    low_prices = s[3::6]       # Low prices are at indices 3, 9, 15, ...
    volumes = s[4::6]          # Volumes are at indices 4, 10, 16, ...
    
    # Feature 1: Price Momentum (current closing price change)
    # Avoid division by zero with np.where
    price_momentum = np.where(opening_prices != 0, (closing_prices - opening_prices) / opening_prices, 0)
    
    # Feature 2: Average Trading Volume over the last 20 days
    avg_volume = np.mean(volumes)
    
    # Feature 3: Rate of Change of Closing Prices over the last 20 days
    # Calculate rate of change
    roc = np.where(closing_prices[:-1] != 0, (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1], 0)
    
    # Return the computed features as a 1D numpy array
    features = [np.mean(price_momentum), avg_volume, np.mean(roc)]
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
        reward += -np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)     # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += -np.random.uniform(5, 15)     # moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 30)  # positive reward for upward features
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 30)  # positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(-10, 10)  # Reward mean-reversion features
        reward += -np.random.uniform(5, 15)    # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is capped within the specified range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward