import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    # Extracting the closing prices for the past 20 days
    closing_prices = s[0:120:6]
    
    # Feature 1: Moving Average (MA) over last 5 days
    ma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    
    # Feature 2: Relative Strength Index (RSI) - 14 days (simplified)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0).mean() if len(deltas) > 0 else 0
    loss = -np.where(deltas < 0, deltas, 0).mean() if len(deltas) > 0 else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 3: Price Change Percentage (over the last 5 days)
    price_change_pct = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100 if len(closing_prices) >= 6 and closing_prices[-6] != 0 else 0
    
    # Collecting features in a list
    features = [ma_5, rsi, price_change_pct]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward = -40  # STRONG NEGATIVE reward for BUY
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -20  # Moderate negative reward for BUY
        
    # If risk level is low, we assess trend and volatility
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                reward += 10  # Positive reward for upward features
            elif trend_direction < -0.3:
                reward += 10  # Positive reward for downward features
        
        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Assuming we have features that indicate mean reversion (like RSI)
            rsi = enhanced_s[123]  # Assuming RSI is the first feature in revised state
            if rsi < 30:  # Oversold condition
                reward += 10  # Reward for potential buy
            elif rsi > 70:  # Overbought condition
                reward += 10  # Reward for potential sell
            
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return reward