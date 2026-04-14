import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes
    
    # Feature 1: Bollinger Bands - Upper and Lower Bands (20-day)
    if len(closing_prices) >= 20:
        sma = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
    else:
        upper_band = lower_band = np.nan
    
    # Feature 2: Average True Range (ATR) - volatility measure
    true_ranges = np.zeros(len(closing_prices) - 1)
    for i in range(1, len(closing_prices)):
        high = s[2::6][i]  # High prices
        low = s[3::6][i]   # Low prices
        true_ranges[i-1] = max(high[i-1], closing_prices[i-1]) - min(low[i-1], closing_prices[i-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.nan
    
    # Feature 3: Positive to Negative Price Change Ratio
    price_changes = np.diff(closing_prices)
    pos_changes = np.sum(price_changes[price_changes > 0])
    neg_changes = np.abs(np.sum(price_changes[price_changes < 0]))
    pos_neg_ratio = pos_changes / (neg_changes + 1e-10)  # Adding a small value to avoid division by zero
    
    features = [upper_band, lower_band, atr, pos_neg_ratio]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds based on the provided state
    historical_std = np.std(enhanced_s[123:])  # Using features for historical std
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std
    trend_threshold = 0.3

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY
        reward += 10   # Mild positive reward for SELL
    elif risk_level > risk_threshold_moderate:
        reward -= 20  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        if trend_direction > trend_threshold:
            reward += np.random.uniform(10, 30)  # Positive reward for upward features
        elif trend_direction < -trend_threshold:
            reward += np.random.uniform(10, 30)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)