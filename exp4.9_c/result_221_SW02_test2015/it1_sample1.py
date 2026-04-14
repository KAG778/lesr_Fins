import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes
    num_days = len(closing_prices)

    # Feature 1: Price Momentum (percentage change)
    if num_days > 1:
        momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        momentum = 0
    features.append(momentum)

    # Feature 2: Average Volume Change (relative change over last 5 days)
    if num_days >= 6:
        avg_volume_last_5 = np.mean(volumes[-5:])
        avg_volume_previous_5 = np.mean(volumes[-10:-5]) if len(volumes) > 10 else avg_volume_last_5
        volume_change = (avg_volume_last_5 - avg_volume_previous_5) / avg_volume_previous_5 if avg_volume_previous_5 != 0 else 0
    else:
        volume_change = 0
    features.append(volume_change)

    # Feature 3: Bollinger Band Width (standard deviation of recent prices)
    if num_days >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        band_width = rolling_std / rolling_mean if rolling_mean != 0 else 0
    else:
        band_width = 0
    features.append(band_width)

    # Feature 4: Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extracting features from enhanced state
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong negative for BUY-aligned features
        reward += 10 if features[0] < 0 else 0  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish momentum
            reward += 15  # Positive reward for following the trend
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish momentum
            reward += 15  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # RSI indicating oversold
            reward += 15  # Reward for buying in oversold condition
        elif features[3] > 70:  # RSI indicating overbought
            reward -= 15  # Penalize for buying in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds