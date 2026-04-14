import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    high_prices = s[2::6]      # High prices
    low_prices = s[3::6]       # Low prices

    # Feature 1: Rate of Change (ROC) over the last 5 days
    roc = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if closing_prices[-5] != 0 else 0
    
    # Feature 2: Average True Range (ATR) for Volatility
    true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:], high_prices[1:] - closing_prices[:-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    
    # Feature 3: Z-score of Volume Change
    volume_change = np.diff(volumes)  # Change in volume
    volume_change_z = (volume_change[-1] - np.mean(volume_change)) / np.std(volume_change) if np.std(volume_change) != 0 else 0

    # Feature 4: Distance from 20-day Moving Average (Mean Reversion)
    moving_average = np.mean(closing_prices[-20:])  # 20-day moving average
    distance_from_ma = closing_prices[-1] - moving_average  # Current price - MA

    features = [roc, atr, volume_change_z, distance_from_ma]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_risk_threshold = np.std(enhanced_s[123:])  # Example for dynamic threshold, adjust as per historical data
    historical_trend_threshold = 0.3 * historical_risk_threshold  # Use a fraction of the risk threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY signals
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL signals
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > historical_trend_threshold and risk_level < 0.4:
        if trend_direction > 0:
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        else:
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < historical_trend_threshold and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]