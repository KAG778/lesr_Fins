import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    num_prices = len(closing_prices)

    # Feature 1: Exponential Moving Average (EMA) for trend detection
    if num_prices >= 14:
        ema = np.mean(closing_prices[-14:])  # 14-day EMA
    else:
        ema = 0
    features.append(ema)

    # Feature 2: Bollinger Bands - Upper and Lower Bands
    if num_prices >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
    else:
        upper_band = lower_band = 0
    features.append(upper_band)
    features.append(lower_band)

    # Feature 3: Average True Range (ATR) for volatility measure
    highs = s[2::6]
    lows = s[3::6]
    true_ranges = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    features.append(atr)

    # Feature 4: Z-score of Daily Returns for mean reversion
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if num_prices > 1 else np.array([0])
    if len(daily_returns) > 0:
        mean_returns = np.mean(daily_returns)
        std_returns = np.std(daily_returns)
        z_score = (daily_returns[-1] - mean_returns) / std_returns if std_returns != 0 else 0
    else:
        z_score = 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    volatility_threshold = np.mean(features[3]) + 2 * np.std(features[3])  # Using ATR as volatility measure
    z_score_threshold = np.mean(features[4])  # Using Z-score as measure

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assuming positive EMA indicates a buy signal
            reward -= 50  # Strong negative for BUY
        else:
            reward += 10  # Mild positive for SELL

    # Priority 2 — TREND FOLLOWING
    elif risk_level < 0.4 and abs(trend_direction) > 0.3:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += 20  # Reward for alignment
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += 20  # Reward for alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3:
        if features[4] < -1:  # Z-score indicating extreme oversold condition
            reward += 15  # Reward for buying in oversold conditions
        elif features[4] > 1:  # Z-score indicating extreme overbought condition
            reward -= 15  # Penalty for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return float(np.clip(reward, -100, 100))