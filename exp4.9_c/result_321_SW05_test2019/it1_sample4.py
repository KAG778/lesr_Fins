import numpy as np

def revise_state(s):
    # s: 120d raw state, where each day has 6 features: [close, open, high, low, volume, adj_close]
    num_days = 20
    closing_prices = s[0:num_days*6:6]  # Extract closing prices
    volumes = s[4:num_days*6:6]          # Extract volumes

    # Feature 1: Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0).mean()
        loss = -np.where(delta < 0, delta, 0).mean()  # Use negative losses
        rs = gain / loss if loss > 0 else 0  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices[-14:])  # Using the last 14 days for RSI

    # Feature 2: Moving Average of closing prices
    sma_10 = np.mean(closing_prices[-10:])  # 10-day simple moving average
    sma_30 = np.mean(closing_prices[-30:]) if len(closing_prices) >= 30 else 0  # 30-day SMA

    # Feature 3: Volume Change Percentage
    avg_volume_last_5 = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 1  # Avoid division by zero
    recent_volume = volumes[-1]
    volume_change_pct = (recent_volume - avg_volume_last_5) / avg_volume_last_5  # Relative change in volume

    # Feature 4: Price Change Percentage (last to previous)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / (closing_prices[-2] if closing_prices[-2] != 0 else 1)

    return np.array([rsi, sma_10, sma_30, volume_change_pct, price_change_pct])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate historical standard deviations for thresholds
    historical_std = np.std(enhanced_s[123:])  # Use features to determine thresholds
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        if enhanced_s[123] > 0:  # Features indicate a BUY
            reward += np.random.uniform(-50, -30)  # Strong negative reward for BUY
        elif enhanced_s[123] < 0:  # Features indicate a SELL
            reward += np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > 0.4 * historical_std:
        if enhanced_s[123] > 0:
            reward += np.random.uniform(-20, -10)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold and enhanced_s[123] > 0:  # Positive trend and BUY signal
            reward += np.random.uniform(10, 20)  # Positive reward for aligning with trend
        elif trend_direction < -trend_threshold and enhanced_s[123] < 0:  # Negative trend and SELL signal
            reward += np.random.uniform(10, 20)  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        if enhanced_s[123] > 0:  # Oversold condition for BUY
            reward += np.random.uniform(10, 20)  # Reward for mean-reversion buy
        elif enhanced_s[123] < 0:  # Overbought condition for SELL
            reward += np.random.uniform(10, 20)  # Reward for mean-reversion sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds