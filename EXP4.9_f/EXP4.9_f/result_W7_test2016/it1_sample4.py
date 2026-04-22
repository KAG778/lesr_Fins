import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Relative Strength Index (RSI)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI when not enough data
    features.append(rsi)

    # Feature 2: Price Range Over the Last 5 Days
    if len(closing_prices) >= 5:
        price_range = np.max(closing_prices[-5:]) - np.min(closing_prices[-5:])
    else:
        price_range = 0
    features.append(price_range)

    # Feature 3: Volume Moving Average (last 10 days)
    if len(volumes) >= 10:
        volume_ma = np.mean(volumes[-10:])
    else:
        volume_ma = np.mean(volumes) if len(volumes) > 0 else 0
    features.append(volume_ma)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[0:120])  # Use the raw state for variability
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 90  # Strong negative reward for BUY-aligned features
        reward += 10 if enhanced_s[123] < 0 else 0  # Mild positive for SELL-aligned features
    elif risk_level > 0.4 * historical_std:
        reward -= 30  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold:
            reward += 20  # Positive reward for upward features
        elif trend_direction < -trend_threshold:
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)

    return reward