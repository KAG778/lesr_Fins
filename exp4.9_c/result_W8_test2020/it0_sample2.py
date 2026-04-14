import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate the price changes
    closing_prices = s[0::6]  # Extract closing prices
    price_change = np.diff(closing_prices)  # Day-to-day price changes
    
    # Feature 1: 20-day moving average
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else np.nan
    
    # Feature 2: Price momentum (last price change)
    last_momentum = price_change[-1] if len(price_change) > 0 else 0.0
    
    # Feature 3: Volatility (standard deviation of closing prices over last 20 days)
    volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else np.nan
    
    # Handle NaN values by replacing with zero
    moving_average = moving_average if not np.isnan(moving_average) else 0.0
    last_momentum = last_momentum if not np.isnan(last_momentum) else 0.0
    volatility = volatility if not np.isnan(volatility) else 0.0
    
    # Collect features
    features.append(moving_average)
    features.append(last_momentum)
    features.append(volatility)
    
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
        reward = -40.0  # Example strong negative reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -20.0  # Example moderate negative reward

    # Only check further priorities if risk is low
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                reward += 10.0  # Positive reward for upward trend
            elif trend_direction < -0.3:
                reward += 10.0  # Positive reward for downward trend

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        if abs(trend_direction) < 0.3:
            # Assume features[0] is a mean-reversion indicator here
            oversold_threshold = -0.02  # Example threshold for oversold
            overbought_threshold = 0.02  # Example threshold for overbought
            # Placeholder checks for buy/sell signals; modify as needed based on features
            if enhanced_state[123] < oversold_threshold:
                reward += 15.0  # Reward for buying an oversold signal
            elif enhanced_state[123] > overbought_threshold:
                reward += 15.0  # Reward for selling an overbought signal

        # Priority 4 — HIGH VOLATILITY
        if volatility_level > 0.6:
            reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return float(np.clip(reward, -100, 100))