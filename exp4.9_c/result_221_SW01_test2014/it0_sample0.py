import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extracting closing prices
    opening_prices = s[1::6]  # Extracting opening prices
    high_prices = s[2::6]     # Extracting high prices
    low_prices = s[3::6]      # Extracting low prices
    volumes = s[4::6]         # Extracting trading volumes
    
    # Feature 1: 14-day Simple Moving Average (SMA) of closing prices
    sma_period = 14
    if len(closing_prices) >= sma_period:
        sma = np.mean(closing_prices[-sma_period:])
    else:
        sma = np.nan  # Handle edge case
    
    # Feature 2: Price volatility (standard deviation) over the last 14 days
    volatility = np.std(closing_prices[-sma_period:]) if len(closing_prices) >= sma_period else np.nan
    
    # Feature 3: Price momentum (current closing price - closing price n days ago)
    price_momentum = closing_prices[-1] - closing_prices[-sma_period] if len(closing_prices) >= sma_period else np.nan
    
    # Feature 4: Daily price change (percentage)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100 if len(closing_prices) > 1 else np.array([np.nan])
    
    # Return only the computed features, filtering out NaN values
    features = [sma, volatility, price_momentum, daily_returns[-1]]  # Including the last daily return
    
    # Ensure all features are valid numbers (replace NaN with 0)
    features = [f if np.isfinite(f) else 0 for f in features]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = computed features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative for BUY-aligned features
        reward += -40  # Example value in the range -30 to -50
    elif risk_level > 0.4:
        # Moderate negative for BUY signals
        reward += -15  # Example value for moderate negative reward

    # If risk is low, consider trend and volatility
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:  # Uptrend
                reward += 20  # Positive reward for bullish features
            elif trend_direction < -0.3:  # Downtrend
                reward += 20  # Positive reward for bearish features

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Assuming we have features indicating overbought/oversold conditions
            # Here we assume features[0] indicates overbought/oversold states
            # For simplicity, let's say features[0] is a momentum value
            momentum = enhanced_state[123]  # Using price momentum from revise_state
            
            if momentum < -2:  # Oversold condition
                reward += 10  # Reward for potential buy
            elif momentum > 2:  # Overbought condition
                reward += -10  # Penalize for potential buy

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range