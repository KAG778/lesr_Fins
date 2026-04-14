import numpy as np

def revise_state(s):
    # Extract closing prices from raw state
    closing_prices = s[0:120:6]  # every 6th element starting from index 0
    volumes = s[4:120:6]          # Extract volumes

    features = []

    # 1. Price Momentum (percentage change over the last 5 days)
    if len(closing_prices) > 5:
        price_momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5]
    else:
        price_momentum = 0
    features.append(price_momentum)

    # 2. Average True Range (ATR) for Volatility
    def calculate_atr(prices, n=14):
        if len(prices) < n:
            return 0
        tr = np.maximum(prices[1:] - prices[:-1], np.maximum(prices[1:] - prices[:-1], prices[:-1] - prices[1:]))
        return np.mean(tr[-n:])

    atr = calculate_atr(closing_prices)
    features.append(atr)

    # 3. Rate of Change of RSI (to identify momentum shifts)
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi_value = calculate_rsi(closing_prices[-14:])
    features.append(rsi_value)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate thresholds based on historical standard deviations
    historical_std = np.std(enhanced_s[0:120])  # Example of using the raw state for std calculation

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(40, 60)  # Strong negative reward for BUY
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 30)  # Positive reward for upward momentum
        else:  # Downtrend
            reward += np.random.uniform(10, 30)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 30:  # Potentially oversold signal
            reward += np.random.uniform(5, 15)  # Reward for buying in oversold conditions
        elif enhanced_s[123] > 70:  # Potentially overbought signal
            reward += np.random.uniform(5, 15)  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds