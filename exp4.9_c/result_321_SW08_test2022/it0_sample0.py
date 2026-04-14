import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Feature 1: Price Momentum (Current Close - Previous Close)
    if s[6] != 0:  # Prevent division by zero
        momentum = (s[6] - s[0]) / s[0]  # Current close vs first close
        features.append(momentum)
    else:
        features.append(0)  # Default value if division by zero

    # Feature 2: Average Volume over the last 20 days
    average_volume = np.mean(s[4::6])
    features.append(average_volume)

    # Feature 3: Price Range (High - Low) of the most recent day
    price_range = s[5] - s[3]
    features.append(price_range)

    # Feature 4: Relative Strength Index (RSI) calculation (14-day)
    closing_prices = s[0::6]  # Extract closing prices
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = -np.where(deltas < 0, deltas, 0).mean()
        
        # Avoid division by zero
        if loss == 0:
            rs = 100
        else:
            rs = gain / loss
        
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)
    else:
        features.append(50)  # Default RSI if not enough data

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
        # Strong negative reward for BUY-aligned features
        reward += -40  # Example strong penalty for buying in dangerous territory
        # Mild positive reward for SELL-aligned features
        reward += 10   # Encouragement to sell
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Example moderate penalty for buying in elevated risk

    # Proceed to next priorities only if risk is low
    elif risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                # Reward for upward features
                reward += 15  # Example positive reward
            elif trend_direction < -0.3:
                # Reward for downward features
                reward += 15  # Example positive reward for bearish bets

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Reward mean-reversion features
            # As an example, let's assume we have features that imply mean reversion
            reward += 10  # Mild positive reward for mean-reversion actions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward