import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    recent_closing_price = closing_prices[-1]  # Most recent closing price
    previous_closing_price = closing_prices[-2]  # Second most recent closing price
    
    # Feature 1: Price Momentum
    price_momentum = recent_closing_price - previous_closing_price
    
    # Feature 2: 5-Day Moving Average
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    
    # Feature 3: Relative Strength Index (RSI) Calculation
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    average_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    average_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0

    rs = average_gain / average_loss if average_loss != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs)) if average_loss > 0 else 100  # RSI calculation
    
    # Combine features into a single array
    features = [price_momentum, moving_average, rsi]
    
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
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)  # Moderate negative for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive for upward features
        else:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    # Ensure reward stays within bounds of [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward