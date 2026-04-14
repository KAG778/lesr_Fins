import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes

    features = []

    # Feature 1: Average True Range (ATR) for volatility measure
    def calculate_atr(highs, lows, period=14):
        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(abs(highs[1:] - lows[:-1]), 
                                   abs(lows[1:] - lows[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    atr_value = calculate_atr(high_prices, low_prices)
    features.append(atr_value)

    # Feature 2: Rate of Change (ROC) of closing prices
    roc = (closing_prices[-1] - closing_prices[-10]) / closing_prices[-10] if len(closing_prices) > 10 else 0
    features.append(roc)

    # Feature 3: Z-Score of Closing Prices (indicates how far prices are from the mean)
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:])
        z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    else:
        z_score = 0
    features.append(z_score)

    # Feature 4: Volume Spike Detection
    if len(volumes) >= 2:
        avg_volume = np.mean(volumes[:-1])
        volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    else:
        volume_change = 0.0
    features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(enhanced_s[123:]) if len(enhanced_s[123:]) > 0 else 1  # Avoid division by zero
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:  # Uptrend
            reward += 20  # Positive reward for upward alignment
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 20  # Positive reward for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Oversold condition
            reward += 15  # Reward for buy
        else:  # Overbought condition
            reward -= 15  # Penalize for sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]