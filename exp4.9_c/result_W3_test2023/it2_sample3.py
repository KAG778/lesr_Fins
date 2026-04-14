import numpy as np

def revise_state(s):
    # Extract closing prices, high, low, and volumes from the raw state
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    features = []

    # Feature 1: 14-day Average True Range (ATR) for volatility measure
    def calculate_atr(highs, lows, period=14):
        true_ranges = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closing_prices[:-1]), abs(lows[1:] - closing_prices[:-1])))
        atr = np.mean(true_ranges[-period:]) if len(true_ranges) >= period else 0
        return atr

    atr_value = calculate_atr(high_prices, low_prices)

    # Feature 2: 14-day Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0).mean() if len(deltas) >= period else 0
        losses = -np.where(deltas < 0, deltas, 0).mean() if len(deltas) >= period else 0
        rs = gains / losses if losses != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_value = calculate_rsi(closing_prices)

    # Feature 3: Rate of Change (ROC) over the last 10 days
    roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) > 10 else 0

    # Feature 4: Z-Score of the last closing price based on historical data
    historical_mean = np.mean(closing_prices) if len(closing_prices) > 0 else 0
    historical_std = np.std(closing_prices) if len(closing_prices) > 0 else 1  # Avoid division by zero
    z_score = (closing_prices[-1] - historical_mean) / historical_std

    features = [atr_value, rsi_value, roc, z_score]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on the historical standard deviation of features
    historical_std = np.std(enhanced_s[123:]) if len(enhanced_s[123:]) > 0 else 1  # Avoid division by zero
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:  # Uptrend
            reward += 10  # Positive reward for upward alignment
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 10  # Positive reward for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if enhanced_s[123] < -0.5:  # Assuming oversold situation
            reward += 15  # Reward mean-reversion BUY
        elif enhanced_s[123] > 0.5:  # Overbought situation
            reward -= 15  # Penalize for selling in a mean-reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]