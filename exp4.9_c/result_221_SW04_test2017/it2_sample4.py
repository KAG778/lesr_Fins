import numpy as np

def revise_state(s):
    # Extracting closing prices and trading volumes
    closing_prices = s[0:120:6]  # Closing prices
    trading_volumes = s[4:120:6]  # Trading volumes

    # Feature 1: 14-day Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = -np.where(deltas < 0, deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 2: 14-day Moving Average
    moving_average = np.mean(closing_prices[-14:]) if len(closing_prices) >= 14 else 0

    # Feature 3: Price Change Ratio (last price to moving average)
    price_change_ratio = (closing_prices[-1] - moving_average) / moving_average if moving_average != 0 else 0  # Prevent division by zero

    # Feature 4: Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0

    # Feature 5: Volume Change Ratio (current volume to average volume)
    avg_volume = np.mean(trading_volumes) if len(trading_volumes) > 0 else 1e-5  # Avoid division by zero
    volume_change_ratio = trading_volumes[-1] / avg_volume if avg_volume > 0 else 0

    features = [rsi, moving_average, price_change_ratio, atr, volume_change_ratio]
    return np.nan_to_num(np.array(features))  # Replace NaNs with 0 to ensure usability

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]

    # Calculate dynamic thresholds based on historical volatility
    historical_std = np.std(features)  # Using the features to assess standard deviation
    risk_threshold = 0.5  # Example threshold for risk level
    high_vol_threshold = 0.6 * historical_std  # Volatility based on historical std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50 * (features[2] > 0)  # Strong negative for BUY signals in high risk
        reward += 10 * (features[2] < 0)  # Mild positive for SELL signals
    elif risk_level > 0.4:
        reward -= 20 * (features[2] > 0)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 20 * (features[2] > 0)  # Positive reward for aligning with upward trend
        reward += 20 * (features[2] < 0)  # Positive reward for aligning with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 30:  # Assuming RSI < 30 indicates oversold
            reward += 15  # Reward for buying in oversold conditions
        if features[0] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > high_vol_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds