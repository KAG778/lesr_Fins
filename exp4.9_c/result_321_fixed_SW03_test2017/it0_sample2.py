import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices are at indices 0, 6, 12, ..., 114 (20 values)
    volumes = s[4::6]          # Volumes are at indices 4, 10, 16, ..., 114 (20 values)
    
    # Feature 1: Price Momentum (Current closing price - Previous closing price)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0.0
    
    # Feature 2: Average Volume Change
    avg_volume_change = np.mean(np.diff(volumes)) if len(volumes) > 1 else 0.0
    
    # Feature 3: Price Range (High - Low over the last 20 days)
    high_prices = s[2::6]    # High prices are at indices 2, 8, 14, ..., 116 (20 values)
    low_prices = s[3::6]      # Low prices are at indices 3, 9, 15, ..., 117 (20 values)
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 else 0.0
    
    features = [price_momentum, avg_volume_change, price_range]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative
        if features[0] > 0:  # If price momentum is positive in a dangerous situation
            reward -= 10.0  # Additional penalty
        return float(np.clip(reward, -100, 100))
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 10.0  # Reward for positive momentum
        else:  # Downtrend
            reward += -features[0] * 10.0  # Reward for negative momentum (correct bearish bet)

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward based on mean-reversion logic (oversold/overbought)
        if features[0] < 0:  # Negative momentum could indicate oversold
            reward += 5.0  # Buy signal
        elif features[0] > 0:  # Positive momentum could indicate overbought
            reward += -5.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))