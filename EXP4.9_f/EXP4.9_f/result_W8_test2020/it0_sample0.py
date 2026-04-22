import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extract closing prices
    volume = s[4::6]  # Extract volume
    
    # Feature 1: Price Momentum (last 5 days)
    price_momentum = closing_prices[-1] - closing_prices[-6]  # Current close - close from 5 days ago
    
    # Feature 2: Relative Strength Index (RSI) (calculate over last 14 days)
    delta = np.diff(closing_prices[-15:])  # Only last 15 days for RSI calculation
    gain = np.where(delta > 0, delta, 0).mean()
    loss = np.abs(np.where(delta < 0, delta, 0)).mean()
    
    # Avoid division by zero
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 3: Volume Change (percentage change over 5 days)
    volume_change = (volume[-1] - volume[-6]) / volume[-6] if volume[-6] != 0 else 0
    
    # Return computed features
    features = [price_momentum, rsi, volume_change]
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
        reward -= 40  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 15  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Assume some features that are aligned with trend (e.g., momentum)
        features = enhanced_state[123:]  # Get features
        if trend_direction > 0.3:  # Uptrend
            reward += max(0, features[0])  # Positive momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += max(0, -features[0])  # Negative momentum (shorting)
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assume features align with mean-reversion (e.g., RSI)
        if enhanced_state[123][1] < 30:  # Oversold
            reward += 10  # Buy signal
        elif enhanced_state[123][1] > 70:  # Overbought
            reward += 10  # Sell signal
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)