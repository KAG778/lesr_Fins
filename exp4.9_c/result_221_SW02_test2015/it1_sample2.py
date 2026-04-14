import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    num_days = len(closing_prices)

    # Feature 1: Price Momentum (percentage change)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Volume Change over the last 5 days (relative change)
    if len(volumes) >= 6:
        avg_volume_last_5 = np.mean(volumes[-5:])
        avg_volume_previous_5 = np.mean(volumes[-10:-5]) if len(volumes) > 10 else avg_volume_last_5
        volume_change = (avg_volume_last_5 - avg_volume_previous_5) / avg_volume_previous_5 if avg_volume_previous_5 != 0 else 0
    else:
        volume_change = 0

    # Feature 3: Bollinger Bands (20-day rolling standard deviation)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        current_price = closing_prices[-1]
        if current_price > upper_band:
            band_signal = 1  # Overbought signal
        elif current_price < lower_band:
            band_signal = -1  # Oversold signal
        else:
            band_signal = 0  # Neutral
    else:
        band_signal = 0

    # Feature 4: Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices) if len(closing_prices) >= 15 else 50  # Default to 50 if insufficient data

    features = [momentum, volume_change, band_signal, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract computed features
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 40  # STRONG NEGATIVE reward for BUY-aligned features
        if features[0] < 0:  # Assuming feature[0] represents a momentum signal suggesting SELL
            reward += 10  # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and positive momentum
            reward += 15  # Positive reward for correct bullish actions
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and negative momentum
            reward += 15  # Positive reward for correct bearish actions

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0:  # Oversold condition
            reward += 15  # Reward for buying in an oversold position
        elif features[2] > 0:  # Overbought condition
            reward -= 15  # Penalize for buying in an overbought position

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensuring reward is within bounds
    return float(np.clip(reward, -100, 100))