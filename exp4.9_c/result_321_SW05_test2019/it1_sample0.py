import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices (every 6th element)
    volumes = s[4:120:6]          # Trading volumes (every 6th element)

    # Calculate Bollinger Bands
    sma = np.mean(closing_prices[-20:])  # Simple moving average of the last 20 days
    std_dev = np.std(closing_prices[-20:])  # Standard deviation of the last 20 days
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)

    # Calculate Average True Range (ATR)
    high_prices = s[2:120:6]  # High prices
    low_prices = s[3:120:6]   # Low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1][1:]), 
                               np.abs(low_prices[1:] - closing_prices[:-1][1:])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Calculate VWAP
    cumulative_vol = np.cumsum(volumes)
    cumulative_vol_price = np.cumsum(closing_prices * volumes)
    vwap = cumulative_vol_price[-1] / cumulative_vol[-1] if cumulative_vol[-1] != 0 else 0

    # Return features: [Bollinger Bands, ATR, VWAP]
    return np.array([upper_band, lower_band, atr, vwap])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds based on statistics
    risk_threshold = 0.7  # Example: 70th percentile of historical risk levels
    trend_threshold = 0.3  # Example: 30th percentile of historical trend levels

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold:
        reward += np.random.uniform(-50, -30)  # Strong negative for BUY
        reward += np.random.uniform(5, 10)  # Mild positive for SELL if selling
    elif risk_level > 0.4:
        reward += np.random.uniform(-20, -10)  # Moderate negative reward for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold:  # Positive trend
            reward += np.random.uniform(15, 25)  # Strong positive for trend-aligned BUY
        elif trend_direction < -trend_threshold:  # Negative trend
            reward += np.random.uniform(15, 25)  # Strong positive for trend-aligned SELL

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion signals

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)