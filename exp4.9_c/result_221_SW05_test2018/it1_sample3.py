import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state.
    
    s: 120d raw state
    Returns ONLY new features (NOT including s or regime).
    """
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes
    
    # Feature 1: Price momentum (current closing price vs. moving average of last 5 days)
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    price_momentum = (closing_prices[-1] - moving_average) / (moving_average if moving_average != 0 else 1)

    # Feature 2: Price range (high - low) over the last 5 days
    price_range = np.max(high_prices[-5:]) - np.min(low_prices[-5:]) if len(high_prices) >= 5 else 0.0

    # Feature 3: Volume change (current volume vs. moving average of last 5 days)
    moving_average_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
    volume_change = (volumes[-1] - moving_average_volume) / (moving_average_volume if moving_average_volume != 0 else 1)

    # Feature 4: Average True Range (ATR) for volatility measure
    true_ranges = np.maximum(high_prices[-5:] - low_prices[-5:], np.abs(closing_prices[-5:] - opening_prices[-5:]))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0

    # Feature 5: Current price relative to the 20-day moving average
    ma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else np.mean(closing_prices)
    price_relative_to_ma = (closing_prices[-1] - ma_20) / (ma_20 if ma_20 != 0 else 1)
    
    return np.array([price_momentum, price_range, volume_change, atr, price_relative_to_ma])

def intrinsic_reward(enhanced_s):
    """
    Computes the reward based on the enhanced state.
    
    enhanced_state[0:120] = raw state
    enhanced_state[120:123] = regime_vector
    enhanced_state[123:] = computed features
    """
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Define relative thresholds based on historical volatility
    risk_threshold_high = 0.7  # This can be adjusted based on backtesting
    risk_threshold_moderate = 0.4
    trend_threshold = 0.3
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Price momentum suggests BUY
            reward -= np.random.uniform(40, 60)  # Strong penalty for BUY
        # Mild positive reward for SELL-aligned features
        elif features[0] < 0:  # Price momentum suggests SELL
            reward += np.random.uniform(10, 20)
    
    elif risk_level > risk_threshold_moderate:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Price momentum suggests BUY
            reward -= np.random.uniform(20, 30)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        if trend_direction > trend_threshold and features[0] > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(20, 30)
        elif trend_direction < -trend_threshold and features[0] < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(20, 30)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition
            reward += np.random.uniform(10, 20)
        elif features[0] > 0.1:  # Overbought condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]