import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Volatility Adjusted Momentum
    if len(closing_prices) > 5:
        daily_returns = np.diff(closing_prices) / closing_prices[:-1]
        mean_return = np.mean(daily_returns)
        volatility = np.std(daily_returns) if len(daily_returns) > 1 else 1e-9  # Avoid division by zero
        momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
        vol_adjusted_momentum = momentum / volatility
    else:
        vol_adjusted_momentum = 0
    features.append(vol_adjusted_momentum)

    # Feature 2: Bollinger Bands (20-day)
    if len(closing_prices) >= 20:
        moving_average = np.mean(closing_prices[-20:])
        upper_band = moving_average + 2 * np.std(closing_prices[-20:])
        lower_band = moving_average - 2 * np.std(closing_prices[-20:])
        current_price = closing_prices[-1]
        bollinger_signal = (current_price - lower_band) / (upper_band - lower_band)  # Normalize to [0, 1]
    else:
        bollinger_signal = 0
    features.append(bollinger_signal)

    # Feature 3: Relative Volume (compared to average)
    if len(volumes) > 20:
        avg_volume = np.mean(volumes[-20:])  # 20-day average volume
        relative_volume = (volumes[-1] / avg_volume) if avg_volume > 0 else 0
    else:
        relative_volume = 0
    features.append(relative_volume)

    # Feature 4: Average True Range (ATR) for volatility measurement
    highs = s[2::6]
    lows = s[3::6]
    if len(highs) > 1 and len(lows) > 1:
        true_ranges = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
        atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    else:
        atr = 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Calculate historical thresholds for dynamic risk evaluation
    historical_std = np.std(features)
    high_risk_threshold = 1.5 * historical_std
    low_risk_threshold = 0.5 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        if features[0] > 0:  # Positive momentum indicates a BUY signal
            reward -= np.random.uniform(40, 60)  # Strong negative for BUY
        else:
            reward += np.random.uniform(5, 15)  # Mild positive for SELL

    elif risk_level > low_risk_threshold:
        if features[0] > 0:  # Positive momentum
            reward -= np.random.uniform(20, 30)  # Moderate negative for BUY

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += np.random.uniform(10, 30)
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += np.random.uniform(10, 30)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 0.2:  # Oversold condition (Bollinger Band)
            reward += np.random.uniform(5, 15)  # Reward for buying in oversold conditions
        elif features[1] > 0.8:  # Overbought condition
            reward += np.random.uniform(5, 15)  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within bounds