import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    volumes = s[4:120:6]          # Trading volumes for 20 days
    days = len(closing_prices)

    # Feature 1: Price Change Percentage (last day closing to previous closing)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # High prices for 20 days
    low_prices = s[3:120:6]   # Low prices for 20 days
    tr = np.maximum(high_prices - low_prices, np.abs(high_prices - np.roll(closing_prices, 1)), np.abs(low_prices - np.roll(closing_prices, 1)))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 3: Price Momentum (compared to the 5-day moving average)
    if days >= 5:
        n_day_avg = np.mean(closing_prices[-5:])
        price_momentum = (closing_prices[-1] - n_day_avg) / n_day_avg if n_day_avg != 0 else 0
    else:
        price_momentum = 0

    # Feature 4: Volume Change Rate (current volume vs average volume)
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    volume_change_rate = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    # Feature 5: Relative Strength Index (RSI) for momentum
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0  # Last 14 days
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0  # Last 14 days

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    features = [price_change_pct, atr, price_momentum, volume_change_rate, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Use features to calculate a standard deviation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # STRONG NEGATIVE for BUY-aligned features
        if trend_direction < 0:  # Mild positive for SELL-aligned features
            reward += 10
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 25 * np.sign(trend_direction)  # Reward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]