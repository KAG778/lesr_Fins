import numpy as np

def revise_state(s):
    # s: 120d raw state
    n_days = 20  # Number of trading days
    
    # Feature 1: Price Change Percentage
    price_change_percentage = [(s[i*6] - s[(i-1)*6]) / s[(i-1)*6] * 100 if i > 0 else 0 
                               for i in range(n_days)]
    
    # Feature 2: Volume Change Percentage
    volume_change_percentage = [(s[i*6 + 4] - s[(i-1)*6 + 4]) / s[(i-1)*6 + 4] * 100 if i > 0 else 0 
                                for i in range(n_days)]
    
    # Feature 3: Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = -np.where(deltas < 0, deltas, 0).mean()
        
        if (gain + loss) == 0:
            return 100  # RSI is neutral if no gains or losses
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_values = [compute_rsi(s[i*6:i*6 + 1]) for i in range(n_days)]
    
    # Stack features into a single array
    features = np.concatenate([price_change_percentage, volume_change_percentage, rsi_values])
    return features

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
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)   # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 15  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20  # Trend is up, reward upward features
        else:
            reward += 20  # Trend is down, reward downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward -= 10  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward