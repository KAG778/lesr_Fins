import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0:120:6]  # Extracting closing prices
    trading_volumes = s[4:120:6]  # Extracting trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) - 20 days
    if len(closing_prices) >= 20:
        weights = np.exp(np.linspace(-1, 0, 20))
        weights /= weights.sum()
        ema_20 = np.dot(weights, closing_prices[-20:])
    else:
        ema_20 = np.nan
    features.append(ema_20)

    # Feature 2: Bollinger Bands (Upper and Lower) - 20 days
    if len(closing_prices) >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
    else:
        upper_band, lower_band = np.nan, np.nan
    features.extend([upper_band, lower_band])

    # Feature 3: 14-day Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = -np.where(delta < 0, delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.nan
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.nan
    rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
    rsi = 100 - (100 / (1 + rs)) if not np.isnan(rs) else np.nan
    features.append(rsi)

    # Feature 4: Price Change Ratio
    price_change_ratio = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) > 1 else np.nan
    features.append(price_change_ratio)

    # Feature 5: Average True Range (ATR)
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    features.append(atr)

    return np.nan_to_num(np.array(features))  # Replace NaNs with 0 to ensure usability

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]

    # Dynamic thresholds based on historical data
    historical_std = np.std(features)  # Using the features to assess standard deviation
    risk_threshold_high = 1.5 * historical_std  # High risk threshold
    risk_threshold_medium = 0.75 * historical_std  # Medium risk threshold
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40 * (features[0] > 0)  # Strong negative for BUY signals in high risk
        reward += 10 * (features[0] < 0)   # Mild positive for SELL signals
    elif risk_level > risk_threshold_medium:
        reward -= 20 * (features[0] > 0)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:
            reward += 20 * features[0]  # Positive reward for upward momentum
        elif trend_direction < 0:
            reward += 20 * -features[0]  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # Assuming RSI < 30 indicates oversold
            reward += 15  # Reward for buying in oversold conditions
        elif features[3] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds