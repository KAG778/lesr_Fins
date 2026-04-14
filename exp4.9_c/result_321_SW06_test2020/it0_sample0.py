import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    features = []
    
    # Extracting the closing prices for the last 20 days
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    
    # Feature 1: Price Change Percentage over the last 5 days
    if len(closing_prices) > 5:
        price_change = ((closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]) * 100
    else:
        price_change = 0  # Handle edge case for insufficient data
    
    features.append(price_change)
    
    # Feature 2: 14-day Relative Strength Index (RSI)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Calculate price changes
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if not enough data
    
    features.append(rsi)
    
    # Feature 3: Exponential Moving Average (EMA) over the last 10 days
    if len(closing_prices) >= 10:
        ema = np.mean(closing_prices[-10:])  # Simplified EMA; typically requires smoothing
    else:
        ema = closing_prices[-1]  # Use latest closing price if not enough data
    
    features.append(ema)
    
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
        # Strong negative reward for BUY-aligned features
        reward = -40  # Example value in the strong negative range
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -20  # Example value in the moderate negative range
    
    # If not in high-risk state, evaluate other priorities
    if risk_level <= 0.4:
        if abs(trend_direction) > 0.3:  # Priority 2 — TREND FOLLOWING
            if trend_direction > 0.3:
                # Reward for upward features (BUY)
                reward += 10  # Positive reward for aligning with uptrend
            elif trend_direction < -0.3:
                # Reward for downward features (SELL)
                reward += 10  # Positive reward for aligning with downtrend
        
        if abs(trend_direction) < 0.3:  # Priority 3 — SIDEWAYS / MEAN REVERSION
            # Checking for mean-reversion features (assuming features are available)
            rsi = enhanced_state[123][1]  # Assuming RSI is the second feature
            if rsi < 30:
                reward += 15  # Oversold condition (suggesting BUY)
            elif rsi > 70:
                reward += 15  # Overbought condition (suggesting SELL)
            
            # Penalize breakout-chasing features (not implemented in this example)

        # Priority 4 — HIGH VOLATILITY
        if volatility_level > 0.6:
            reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)