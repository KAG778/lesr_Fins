import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices are at indices 0, 6, 12, ..., 114 (20 days)
    
    # Feature 1: Price Momentum
    # (Current closing price - Previous closing price) / Previous closing price
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / (closing_prices[-2] if closing_prices[-2] != 0 else 1)
    
    # Feature 2: Simple Moving Average (SMA) of the last 5 days
    sma = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    
    # Feature 3: Relative Strength Index (RSI)
    # Calculate RSI using the last 14 days
    deltas = np.diff(closing_prices[-14:])  # Daily price changes
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if (gain + loss) != 0 else 0
    
    features = [price_momentum, sma, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        return reward  # Early exit
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for uptrend features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downtrend features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward