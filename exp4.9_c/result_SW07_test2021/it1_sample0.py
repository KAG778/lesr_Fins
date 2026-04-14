import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    
    # Feature 1: Price Momentum (5-day change)
    price_momentum = closing_prices[-1] - closing_prices[-5]  # Change from day 15 to day 19

    # Feature 2: RSI (Relative Strength Index)
    gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
    losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 3: Average True Range (ATR)
    high_low = high_prices[-20:] - low_prices[-20:]
    high_close = np.abs(high_prices[-20:] - closing_prices[-21:-1])
    low_close = np.abs(low_prices[-20:] - closing_prices[-21:-1])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = np.mean(tr) if len(tr) > 0 else 0

    # Feature 4: VWAP (Volume Weighted Average Price)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else 0

    # Return the computed features as a numpy array
    return np.array([price_momentum, rsi, atr, vwap])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY signals
        reward += np.random.uniform(5, 10)   # MILD POSITIVE reward for SELL signals
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for bullish features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        rsi = enhanced_s[123]  # Assuming RSI is part of the features
        if rsi < 30:
            reward += 15  # Reward for oversold conditions
        elif rsi > 70:
            reward -= 15  # Penalize for overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)