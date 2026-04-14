import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes
    num_days = len(closing_prices)

    # Feature 1: Relative Momentum (normalized momentum)
    if num_days > 5:
        momentum = (closing_prices[-1] - closing_prices[-6]) / np.std(closing_prices[-6:])
        features.append(momentum)
    else:
        features.append(0)

    # Feature 2: Volume Change (normalized)
    if len(volumes) > 1 and volumes[-2] > 0:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
        features.append(volume_change)
    else:
        features.append(0)

    # Feature 3: Relative Strength Index (RSI)
    if num_days >= 14:  # Need at least 14 days for RSI calculation
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)
    else:
        features.append(0)

    # Feature 4: Average True Range (ATR) for volatility
    highs = s[2::6]
    lows = s[3::6]
    true_ranges = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Relative thresholds for risk evaluation
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    trend_threshold = 0.3

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        reward -= 50 if features[0] > 0 else 0  # BUY signal
        # Mild positive reward for SELL-aligned features
        reward += 10 if features[0] < 0 else 0  # SELL signal
    elif risk_level > risk_threshold_medium:
        # Moderate negative reward for BUY signals
        reward -= 20 if features[0] > 0 else 0

    # Priority 2: Trend Following (when risk is low)
    if risk_level < risk_threshold_medium:
        if abs(trend_direction) > trend_threshold:
            if trend_direction > trend_threshold and features[0] > 0:  # Uptrend and positive momentum
                reward += 20
            elif trend_direction < -trend_threshold and features[0] < 0:  # Downtrend and negative momentum
                reward += 20

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[2] < 30:  # RSI < 30 is considered oversold
            reward += 15  # Reward for buying in oversold conditions
        elif features[2] > 70:  # RSI > 70 is considered overbought
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return max(-100, min(100, reward))