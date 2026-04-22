import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    # Extract the closing prices from the raw state
    closing_prices = s[::6]  # Every 6th element starting from index 0
    opening_prices = s[1::6]  # Every 6th element starting from index 1
    
    # Feature 1: Price Momentum (percentage change from opening to closing)
    price_momentum = (closing_prices - opening_prices) / opening_prices
    price_momentum = np.nan_to_num(price_momentum)  # Handle NaN values
    
    # Feature 2: Simple Moving Average (SMA) of closing prices over the last 5 days
    sma_5 = np.convolve(closing_prices, np.ones(5)/5, mode='valid')
    sma_5 = np.pad(sma_5, (4, 0), 'edge')  # Pad to maintain the same length as closing_prices
    
    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
    
    # To avoid division by zero in RSI calculation
    rs = np.nan_to_num(avg_gain / (avg_loss + 1e-10))
    rsi = 100 - (100 / (1 + rs))
    rsi = np.pad(rsi, (13, 0), 'edge')  # Pad to maintain the same length as closing_prices

    # Combine features into a single array
    features = [price_momentum[-20:], sma_5[-20:], rsi[-20:]]  # Keep only the last 20 values
    return np.concatenate(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize the reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward = np.random.uniform(-20, -10)  # Moderate negative reward for BUY signals

    # Check if we are in a risk-free environment to evaluate trends
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:  # Uptrend
                reward += 10  # Positive reward for upward features
            else:  # Downtrend
                reward += 10  # Positive reward for downward features

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Here we would check features related to mean-reversion (not defined in the prompt)
            # This part is left abstract, as it requires specific features to evaluate
            reward += 5  # Example positive reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward