import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (day i at index 6*i)
    volumes = s[4:120:6]          # Extract volumes (day i at index 6*i + 4)

    # Feature 1: Price Momentum (most recent closing price - closing price n days ago)
    # We will use 5 days ago for momentum calculation
    momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Volume Change (percentage change from previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0

    # Feature 3: Price Volatility (standard deviation of the last 20 closing prices)
    price_volatility = np.std(closing_prices) if len(closing_prices) > 0 else 0

    # Return the features as a 1D numpy array
    features = [momentum, volume_change, price_volatility]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        # Seller reward can be mild positive
        reward += np.random.uniform(5, 10)    # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for bullish alignment
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for bearish alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming we have oversold and overbought features in enhanced_state[123:]
        oversold_feature = enhanced_state[123][0]  # Assuming first feature is oversold
        overbought_feature = enhanced_state[123][1]  # Assuming second feature is overbought
        if oversold_feature > 0:  # Example condition for oversold
            reward += 10  # Reward for buying on oversold condition
        elif overbought_feature > 0:  # Example condition for overbought
            reward -= 10  # Penalize for chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% for uncertainty

    return float(reward)