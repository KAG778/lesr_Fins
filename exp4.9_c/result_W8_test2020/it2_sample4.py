import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices

    # 1. Rate of Change (ROC) - measures the speed of price changes
    roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 and closing_prices[-2] != 0 else 0.0
    features.append(roc)

    # 2. Average True Range (ATR) for measuring market volatility
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0
    features.append(atr)

    # 3. Z-Score of daily returns for mean-reversion signals
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0.0
    std_return = np.std(daily_returns) if len(daily_returns) > 0 else 1.0  # Avoid division by zero
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score)

    # 4. Bollinger Bands normalized indicator
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        band_width = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0
    else:
        band_width = 0
    features.append(band_width)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Use computed features for dynamic thresholds
    risk_high_threshold = 0.7 * historical_std
    risk_moderate_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1: Risk Management
    if risk_level > risk_high_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY actions
    elif risk_level > risk_moderate_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY actions

    # Priority 2: Trend Following (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level <= risk_moderate_threshold:
        if trend_direction > trend_threshold:  # Uptrend
            reward += np.random.uniform(10, 20)  # Reward for BUY signals in uptrend
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += np.random.uniform(10, 20)  # Reward for SELL signals in downtrend

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < risk_moderate_threshold:
        if enhanced_s[123] < 0:  # Assuming negative Z-Score indicates oversold
            reward += 15.0  # Reward for buying an oversold signal
        elif enhanced_s[123] > 0:  # Assuming positive Z-Score indicates overbought
            reward += 15.0  # Reward for selling an overbought signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_moderate_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]