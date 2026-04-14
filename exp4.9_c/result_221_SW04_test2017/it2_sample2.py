import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    trading_volumes = s[4:120:6]  # Extract trading volumes
    days = len(closing_prices)

    # Feature 1: 14-day Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = -np.where(deltas < 0, deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 2: Price Change Ratio (current price - previous price) / previous price
    price_change_ratio = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if days > 1 else 0

    # Feature 3: Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0

    # Feature 4: Z-score of Volume compared to the last 20 days
    if len(trading_volumes) >= 20:
        avg_volume = np.mean(trading_volumes[-20:])
        volume_std = np.std(trading_volumes[-20:])
        volume_zscore = (trading_volumes[-1] - avg_volume) / volume_std if volume_std != 0 else 0
    else:
        volume_zscore = 0

    features = [rsi, price_change_ratio, atr, volume_zscore]
    return np.nan_to_num(np.array(features))  # Ensure no NaN values

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0

    # Calculate dynamic thresholds based on historical std
    historical_std = np.std(features)  # Using features to assess standard deviation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_low = 0.4 * historical_std
    volatility_threshold_high = 0.6 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50 * (features[1] > 0)  # Strong negative for buying in high risk
        reward += 10 * (features[1] < 0)   # Mild positive for selling in high risk
    elif risk_level > risk_threshold_low:
        reward -= 20 * (features[1] > 0)  # Moderate negative for buying

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < risk_threshold_low:
        reward += 20 * (features[1] > 0)  # Positive reward for aligning with upward momentum
        reward += 10 * (features[1] < 0)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3:
        if features[0] < 30:  # Oversold condition
            reward += 15  # Reward for buying in oversold conditions
        if features[0] > 70:  # Overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold_high:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds