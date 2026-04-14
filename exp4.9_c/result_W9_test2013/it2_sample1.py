import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Price Change Percentage over the last 10 days
    if len(closing_prices) >= 10:
        price_change_pct = (closing_prices[-1] - closing_prices[-10]) / closing_prices[-10] if closing_prices[-10] != 0 else 0
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # Feature 2: Average True Range (ATR) to measure volatility
    if len(closing_prices) >= 14:
        true_ranges = np.maximum(closing_prices[1:] - closing_prices[:-1], 
                                  np.maximum(closing_prices[1:] - closing_prices[:-1], 
                                             closing_prices[:-1] - closing_prices[1:]))
        atr = np.mean(true_ranges[-14:])  # ATR over the last 14 days
    else:
        atr = 0
    features.append(atr)

    # Feature 3: Bollinger Bands (Lower Band) based on last 20 days
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        lower_band = rolling_mean - (rolling_std * 2)  # 2 standard deviations
    else:
        lower_band = closing_prices[-1] if len(closing_prices) > 0 else 0
    features.append(lower_band)

    # Feature 4: Volume Change Percentage (current vs. average of last 10 days)
    if len(volumes) >= 10:
        avg_volume = np.mean(volumes[-10:])
        volume_change_pct = (volumes[-1] - avg_volume) / avg_volume if avg_volume > 0 else 0
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
    
    # Calculate thresholds based on historical data (using features)
    historical_std = np.std(features)
    historical_mean = np.mean(features)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # If price change percentage is positive (indicating bullish)
            reward -= (30 + 20 * (historical_std))  # Strong negative for BUY
        else:
            reward += (5 + 5 * (historical_std))  # Mild positive for SELL

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish signal
            reward += (20 + 10 * (historical_std))  # Positive reward for bullish momentum
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish signal
            reward += (20 + 10 * (historical_std))  # Positive reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < historical_mean - historical_std:  # Oversold condition
            reward += 15  # Reward for buying in oversold conditions
        elif features[0] > historical_mean + historical_std:  # Overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(np.clip(reward, -100, 100))