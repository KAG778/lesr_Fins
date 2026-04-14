import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices

    # 1. Exponential Moving Average (EMA) for trend detection
    ema_short = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0.0
    ema_long = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0.0
    ema = ema_short - ema_long  # MACD-like feature for trend
    features.append(ema)

    # 2. Average True Range (ATR) for volatility measurement
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0  # 14-day ATR
    features.append(atr)

    # 3. Z-Score of daily returns for relative performance
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0.0
    std_return = np.std(daily_returns) if len(daily_returns) > 0 else 1.0  # Avoid division by zero
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score)

    # 4. Momentum Indicator as Rate of Change (ROC)
    if len(closing_prices) >= 2:
        roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        roc = 0.0
    features.append(roc)

    # 5. Bollinger Bands width for mean reversion signals
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        bandwidth = (upper_band - lower_band) / rolling_mean if rolling_mean != 0 else 0
    else:
        bandwidth = 0
    features.append(bandwidth)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Using features for dynamic thresholds
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY actions
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(10, 20)  # Mild negative for BUY actions

    # Priority 2: Trend Following (when risk is low)
    if abs(trend_direction) > trend_threshold and risk_level <= risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward trend
        elif trend_direction < 0:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < risk_threshold_moderate:
        # Reward for mean-reversion features based on Z-score
        if enhanced_s[123] < -1:  # Oversold condition
            reward += 15.0  # Reward for buying an oversold signal
        elif enhanced_s[123] > 1:  # Overbought condition
            reward += 15.0  # Reward for selling an overbought signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level <= risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50% during high volatility

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within [-100, 100]