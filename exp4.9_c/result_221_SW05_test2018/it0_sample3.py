import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state.
    
    s: 120d raw state
    Returns ONLY new features (NOT including s or regime).
    """
    closing_prices = s[0::6]  # Extract closing prices (indices 0, 6, 12, ...)
    opening_prices = s[1::6]  # Extract opening prices (indices 1, 7, 13, ...)
    high_prices = s[2::6]     # Extract high prices (indices 2, 8, 14, ...)
    low_prices = s[3::6]      # Extract low prices (indices 3, 9, 15, ...)
    volumes = s[4::6]         # Extract trading volumes (indices 4, 10, 16, ...)
    
    # Feature 1: Price momentum (current closing price vs. moving average of last 5 days)
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    price_momentum = (closing_prices[-1] - moving_average) / (moving_average if moving_average != 0 else 1)  # Avoid division by zero

    # Feature 2: Price range (high - low) over the last 5 days
    price_range = np.max(high_prices[-5:]) - np.min(low_prices[-5:]) if len(high_prices) >= 5 else 0.0

    # Feature 3: Volume change (current volume vs. moving average of last 5 days)
    moving_average_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
    volume_change = (volumes[-1] - moving_average_volume) / (moving_average_volume if moving_average_volume != 0 else 1)  # Avoid division by zero

    return np.array([price_momentum, price_range, volume_change])

def intrinsic_reward(enhanced_state):
    """
    Computes the reward based on the enhanced state.
    
    enhanced_state[0:120] = raw state
    enhanced_state[120:123] = regime_vector
    enhanced_state[123:] = computed features
    """
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # price_momentum indicative of a BUY
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # price_momentum indicative of a SELL
            reward += np.random.uniform(5, 10)
    
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # price_momentum indicative of a BUY
            reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Overbought and bearish signal
            reward += np.random.uniform(10, 20)
        elif features[0] > 0:  # Oversold and bullish signal
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)