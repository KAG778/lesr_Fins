import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Average True Range (ATR) for volatility
    def compute_atr(prices, window=14):
        if len(prices) < window:
            return 0.0
        high = prices[1::6]  # High prices (assuming they are every 6th element)
        low = prices[2::6]   # Low prices (assuming they are every 6th element)
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(abs(high[1:] - closing_prices[1::6][:-1]), 
                                   abs(low[1:] - closing_prices[1::6][:-1])))
        atr = np.mean(tr[-window:]) if len(tr) >= window else np.mean(tr)
        return atr

    atr = compute_atr(s)

    # Feature 2: Price Momentum (normalized)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(price_momentum)

    # Feature 3: Modified RSI (with adaptive thresholds based on historical standard deviation)
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return 50  # Neutral RSI
        delta = np.diff(prices)
        gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
        loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = compute_rsi(closing_prices)
    features.append(rsi)

    # Feature 4: Volume Change (percentage change from last day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
    features.append(volume_change)

    # Feature 5: Standard Deviation of Closing Prices (to gauge volatility)
    std_dev = np.std(closing_prices)
    features.append(std_dev)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -50  # Strong negative for BUY
        reward += 10 * features[2]  # Positive for strong volume decrease (feature 2)
    elif risk_level > 0.4:
        reward += -20  # Mild negative for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10 * features[0]  # Reward based on momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 10 * -features[0]  # Negative reward based on momentum for bearish signals

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Assuming RSI < 30 indicates oversold
            reward += 10  # Reward for potential BUY signal
        elif features[1] > 70:  # Assuming RSI > 70 indicates overbought
            reward += -10  # Penalize potential SELL signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > features[4]:  # Compare against standard deviation feature
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds