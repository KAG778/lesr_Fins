import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Price Change Percentage over the last 10 days
    if len(closing_prices) >= 10:
        price_change_pct = (closing_prices[-1] - closing_prices[-10]) / closing_prices[-10] if closing_prices[-10] != 0 else 0
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # Feature 2: Average True Range (ATR) to measure volatility
    if len(closing_prices) >= 14:
        true_ranges = np.abs(closing_prices[1:] - closing_prices[:-1])
        atr = np.mean(true_ranges[-14:])  # ATR over the last 14 days
    else:
        atr = 0
    features.append(atr)

    # Feature 3: RSI-based feature to detect overbought/oversold conditions
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gains = (deltas[deltas > 0]).mean() if np.any(deltas > 0) else 0
        losses = (-deltas[deltas < 0]).mean() if np.any(deltas < 0) else 0
        rs = gains / losses if losses != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Default RSI if not enough data
    features.append(rsi)

    # Feature 4: Volume Change Percentage over the last 10 days
    if len(volumes) >= 10:
        volume_change_pct = (volumes[-1] - np.mean(volumes[-10:])) / np.mean(volumes[-10:]) if np.mean(volumes[-10:]) != 0 else 0
    else:
        volume_change_pct = 0
    features.append(volume_change_pct)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # The new features added
    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_std = np.std(features)
    historical_mean = np.mean(features)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive price change
            reward -= np.clip(50 * (features[0] / historical_mean), 30, 50)  # Strong negative for BUY
        else:
            reward += np.clip(10 * (1 + features[0]), 5, 10)  # Mild positive for SELL

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend
            reward += np.clip(30 * (features[0] / historical_mean), 20, 30)  # Strong positive
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend
            reward += np.clip(30 * (features[0] / historical_mean), 20, 30)  # Strong positive

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Assuming RSI < 30 indicates oversold
            reward += 15  # Reward for buying in oversold
        elif features[2] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 15  # Reward for selling in overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(np.clip(reward, -100, 100))