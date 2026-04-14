import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    high_prices = s[2::6]      # Extract high prices
    low_prices = s[3::6]       # Extract low prices

    # Feature 1: Rate of Change (ROC) over the last 5 days
    roc = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if closing_prices[-5] != 0 else 0

    # Feature 2: Average True Range (ATR) for volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR

    # Feature 3: Z-score of Volume Change
    volume_change = np.diff(volumes)  # Change in volume
    volume_change_z = (volume_change[-1] - np.mean(volume_change)) / np.std(volume_change) if np.std(volume_change) != 0 else 0

    # Feature 4: Distance from the 20-day Moving Average (mean reversion)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    distance_from_ma = closing_prices[-1] - moving_average  # Current price - MA

    # Combine features into a single array
    features = [roc, atr, volume_change_z, distance_from_ma]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data (using a rolling window approach)
    risk_threshold = 0.7 * np.std(enhanced_s[123:])  # Placeholder for historical risk threshold
    trend_threshold = 0.3 * np.std(enhanced_s[123:])  # Placeholder for historical trend threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(40, 60)  # Strong negative reward for BUY signals
        reward += np.random.uniform(5, 15)    # Mild positive reward for SELL signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold:  # Bullish
            reward += np.random.uniform(15, 25)  # Strong positive reward for bullish features
        elif trend_direction < -trend_threshold:  # Bearish
            reward += np.random.uniform(15, 25)  # Strong positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(10, 20)  # Positive reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds