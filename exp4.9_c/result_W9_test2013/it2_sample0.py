import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Price Change Percentage (last 10 days)
    if len(closing_prices) >= 10:
        price_change_pct = (closing_prices[-1] - closing_prices[-10]) / closing_prices[-10] if closing_prices[-10] != 0 else 0
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # Feature 2: Average True Range (ATR) to measure volatility
    if len(closing_prices) >= 14:
        true_ranges = np.maximum(closing_prices[1:] - closing_prices[:-1], 
                                  np.maximum(closing_prices[:-1] - closing_prices[1:], 
                                             closing_prices[1:] - closing_prices[:-1]))
        atr = np.mean(true_ranges[-14:])  # ATR over the last 14 days
    else:
        atr = 0
    features.append(atr)

    # Feature 3: Moving Average Convergence Divergence (MACD)
    if len(closing_prices) >= 26:
        short_ma = np.mean(closing_prices[-12:])  # 12-day EMA
        long_ma = np.mean(closing_prices[-26:])   # 26-day EMA
        macd = short_ma - long_ma
    else:
        macd = 0
    features.append(macd)

    # Feature 4: Volume Spike (current volume compared to the average volume of last 10 days)
    if len(volumes) >= 10:
        avg_volume_10d = np.mean(volumes[-10:])
        volume_spike = (volumes[-1] - avg_volume_10d) / avg_volume_10d if avg_volume_10d > 0 else 0
    else:
        volume_spike = 0
    features.append(volume_spike)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # The new features added
    reward = 0.0

    # Calculate historical thresholds based on feature statistics
    historical_std = np.std(features)
    historical_mean = np.mean(features)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive price change
            reward -= np.random.uniform(30, 50)  # Strong negative for BUY
        else:
            reward += np.random.uniform(5, 10)    # Mild positive for SELL

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish signal
            reward += 20  # Positive reward for bullish momentum
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish signal
            reward += 20  # Positive reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition based on price change percentage
            reward += 15  # Reward for buying in oversold conditions
        elif features[0] > 0.1:  # Overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(np.clip(reward, -100, 100))