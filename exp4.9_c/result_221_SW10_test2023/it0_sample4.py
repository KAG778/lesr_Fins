import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate returns
    closing_prices = s[0::6]  # Extract closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    features.append(np.mean(daily_returns))  # Mean return
    
    # Calculate volatility (standard deviation of returns)
    if len(daily_returns) > 0:
        volatility = np.std(daily_returns)
    else:
        volatility = 0  # Handle edge case
    
    features.append(volatility)  # Volatility feature
    
    # Calculate momentum (current price vs. price n days ago)
    n_days = 5  # Look-back period
    if len(closing_prices) > n_days:
        momentum = (closing_prices[-1] - closing_prices[-n_days]) / closing_prices[-n_days]
    else:
        momentum = 0  # Handle edge case
    
    features.append(momentum)  # Momentum feature
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Reward for upward features
            reward += np.random.uniform(10, 20)  # Example for upward market
        elif trend_direction < -0.3:
            # Reward for downward features
            reward += np.random.uniform(10, 20)  # Example for downward market
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold → buy, overbought → sell)
        reward += np.random.uniform(5, 15)  # Example for mean-reversion
    
    # Penalize breakout-chasing features (not implementing specific breakout logic here)
    
    # Priority 4 — HIGH VOLATILITY (no crisis)
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)