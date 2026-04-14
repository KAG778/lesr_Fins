import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]  # Extracting trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) over 10 days
    ema = np.zeros(20)
    alpha = 2 / (10 + 1)  # Smoothing factor for 10-day EMA
    ema[0] = closing_prices[0]  # Start with the first closing price
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    features.append(ema[-1])  # Most recent EMA
    
    # Feature 2: Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss else np.inf  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # Calculate RSI
    features.append(rsi)  # Current RSI value
    
    # Feature 3: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0
    features.append(vwap)  # Most recent VWAP
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Your computed features from revise_state
    reward = 0.0

    # Define relative thresholds based on historical std dev
    price_changes = features[0]  # EMA
    rsi = features[1]  # RSI
    vwap = features[2]  # VWAP

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        if price_changes < vwap:  # If price is below VWAP, consider selling
            reward += np.random.uniform(10, 20)  # Mild positive reward for SELL
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 10 if price_changes > vwap else 0  # Positive reward if price is above VWAP
        else:  # Downtrend
            reward += 10 if price_changes < vwap else 0  # Positive reward if price is below VWAP

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold condition
            reward += 15  # Reward for mean-reversion buy signal
        elif rsi > 70:  # Overbought condition
            reward -= 15  # Penalty for mean-reversion sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits