import numpy as np

def revise_state(s):
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

    # Feature 2: 14-day Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0

    # Feature 3: Price Momentum (current price - moving average)
    moving_average = np.mean(closing_prices[-20:]) if days >= 20 else 0
    price_momentum = closing_prices[-1] - moving_average if moving_average != 0 else 0

    # Feature 4: Price to Moving Average Ratio (20-day MA)
    price_to_ma_ratio = closing_prices[-1] / moving_average if moving_average != 0 else 0
    
    # Feature 5: Average Trading Volume over the last 20 days
    average_volume = np.mean(trading_volumes[-20:]) if len(trading_volumes) >= 20 else 0

    features = [rsi, atr, price_momentum, price_to_ma_ratio, average_volume]
    return np.nan_to_num(np.array(features))  # Ensure no NaN values

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(features)  # Using all features for thresholding
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    volatility_threshold_high = 0.6 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50 * (features[0] > 70)  # Strong negative for high-risk buy signals (RSI overbought)
        reward += 10 * (features[0] < 30)  # Mild positive for selling in oversold conditions (RSI undersold)
    elif risk_level > risk_threshold_medium:
        reward -= 20 * (features[0] > 70)  # Moderate negative for high-risk buy signals (RSI overbought)

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += 20 * features[2]  # Reward momentum alignment in an uptrend
        elif trend_direction < 0:  # Downtrend
            reward += 20 * -features[2]  # Reward momentum alignment in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 30:  # Oversold
            reward += 15  # Reward for buying in oversold conditions
        if features[0] > 70:  # Overbought
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold_high and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds