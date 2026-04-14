import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices of the last 20 days
    high_prices = s[2::6]      # High prices of the last 20 days
    low_prices = s[3::6]       # Low prices of the last 20 days
    volumes = s[4::6]          # Trading volumes of the last 20 days

    # Feature 1: Bollinger Bands
    window = 20
    if len(closing_prices) >= window:
        moving_avg = np.mean(closing_prices[-window:])
        std_dev = np.std(closing_prices[-window:])
        upper_band = moving_avg + (2 * std_dev)
        lower_band = moving_avg - (2 * std_dev)
        price = closing_prices[-1]
        bollinger_feature = (price - moving_avg) / std_dev if std_dev != 0 else 0
    else:
        bollinger_feature = 0

    # Feature 2: Exponential Moving Average (EMA)
    alpha = 2 / (window + 1)
    ema = np.array([closing_prices[0]] if closing_prices else [0])
    for price in closing_prices[1:]:
        ema = np.append(ema, (price - ema[-1]) * alpha + ema[-1])
    ema_feature = (closing_prices[-1] - ema[-1]) / ema[-1] if ema[-1] != 0 else 0

    # Feature 3: Average True Range (ATR)
    tr = np.maximum(high_prices[-1] - low_prices[-1], high_prices[-1] - closing_prices[-2], closing_prices[-2] - low_prices[-1])
    atr = np.mean(tr) if len(tr) > 0 else 0  # Simple ATR; could be made more complex
    atr_feature = atr / closing_prices[-1] if closing_prices[-1] != 0 else 0

    return np.array([bollinger_feature, ema_feature, atr_feature])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Compute historical mean and standard deviation for relative thresholds
    historical_values = enhanced_s[123:]  # Assuming historical values are passed
    mean_risk = np.mean(historical_values)
    std_risk = np.std(historical_values)
    upper_threshold = mean_risk + 2 * std_risk
    lower_threshold = mean_risk - 2 * std_risk

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > upper_threshold:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)
        # MILD positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10)
    elif risk_level > lower_threshold:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(5, 15)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < lower_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for bullish features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < lower_threshold:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += 10  # Example positive reward for mean-reversion alignment

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < lower_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Constrain reward within [-100, 100] if necessary
    reward = np.clip(reward, -100, 100)

    return reward