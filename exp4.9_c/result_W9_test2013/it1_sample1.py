import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes
    
    # Feature 1: Moving Average Convergence Divergence (MACD)
    if len(closing_prices) >= 26:
        short_ema = np.mean(closing_prices[-12:])  # Short EMA (12 days)
        long_ema = np.mean(closing_prices[-26:])   # Long EMA (26 days)
        macd = short_ema - long_ema
    else:
        macd = 0
    features.append(macd)

    # Feature 2: Average True Range (ATR) for volatility
    if len(closing_prices) >= 14:
        highs = s[2::6]  # High prices
        lows = s[3::6]   # Low prices
        tr = np.maximum(highs[-1] - lows[-1], np.maximum(abs(highs[-1] - closing_prices[-2]), abs(lows[-1] - closing_prices[-2])))
        atr = np.mean(tr[-14:])  # Average of True Ranges over 14 days
    else:
        atr = 0
    features.append(atr)

    # Feature 3: Stochastic Oscillator
    if len(closing_prices) >= 14:
        lowest_low = np.min(closing_prices[-14:])
        highest_high = np.max(closing_prices[-14:])
        k_value = (closing_prices[-1] - lowest_low) / (highest_high - lowest_low) * 100 if highest_high != lowest_low else 0
    else:
        k_value = 0
    features.append(k_value)

    # Feature 4: Volume Weighted Average Price (VWAP)
    if len(volumes) >= 14:
        vwap = np.sum(closing_prices[-14:] * volumes[-14:]) / np.sum(volumes[-14:]) if np.sum(volumes[-14:]) != 0 else 0
    else:
        vwap = closing_prices[-1]  # If not enough data, fallback to last price
    features.append(vwap)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # The new features added
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # If MACD is positive (suggests bullish)
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # If MACD is negative (suggests bearish)
            reward += np.random.uniform(5, 10)

    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= 15  # Arbitrary value for moderate penalty

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish
            reward += 20  # Reward for correct bullish signal
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish
            reward += 20  # Reward for correct bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 20:  # Assuming Stochastic Oscillator < 20 indicates oversold
            reward += 15  # Reward for buying in oversold
        elif features[2] > 80:  # Assuming Stochastic Oscillator > 80 indicates overbought
            reward += 15  # Reward for selling in overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    # Ensure reward is within [-100, 100]
    return float(np.clip(reward, -100, 100))