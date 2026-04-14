import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices (every 6th element starting from index 0)
    volumes = s[4:120:6]          # Trading volumes (every 6th element starting from index 4)

    # Feature 1: Average True Range (ATR) for volatility measurement
    def calculate_atr(prices, period=14):
        high = np.array([prices[i] for i in range(1, len(prices))])
        low = np.array([prices[i] for i in range(1, len(prices))])
        close = np.array([prices[i] for i in range(len(prices) - 1)])
        tr = np.maximum(high - low, np.maximum(abs(high - close[:-1]), abs(low - close[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    atr = calculate_atr(closing_prices)

    # Feature 2: Bollinger Bands (20-day moving average and standard deviation)
    ma20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    stddev20 = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    upper_band = ma20 + (2 * stddev20)
    lower_band = ma20 - (2 * stddev20)

    # Feature 3: Percentage of Price Above Lower Band
    price_position = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0

    # Return the computed features
    return np.array([atr, ma20, price_position])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical volatility
    historical_volatility = np.std(enhanced_s[0:120:6])  # Historical closing price volatility
    risk_threshold_high = historical_volatility * 1.5
    risk_threshold_mid = historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if trend_direction > 0:  # Strong BUY signal
            reward -= np.random.uniform(30, 50)  # Strong negative reward for risky BUY
        else:  # Strong SELL signal
            reward += np.random.uniform(10, 20)  # Mild positive reward for risky SELL
    elif risk_level > 0.4:
        if trend_direction > 0:
            reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 30:  # Assuming a feature indicates oversold condition
            reward += 10  # Reward for mean-reversion BUY
        elif enhanced_s[123] > 70:  # Assuming a feature indicates overbought condition
            reward += 10  # Reward for mean-reversion SELL

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)