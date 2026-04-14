import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Change Percentage (last day vs previous day)
    closing_prices = s[0::6]
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_change_pct)

    # Feature 2: Moving Average Convergence Divergence (MACD) - to capture trend changes
    if len(closing_prices) >= 26:
        ema_12 = np.mean(closing_prices[-12:])  # 12-day EMA
        ema_26 = np.mean(closing_prices[-26:])  # 26-day EMA
        macd = ema_12 - ema_26
    else:
        macd = 0
    features.append(macd)

    # Feature 3: Average True Range (ATR) for volatility measurement
    highs = s[2::6]
    lows = s[3::6]
    tr = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[1:-1], closing_prices[1:-1] - lows[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    features.append(atr)

    # Feature 4: 14-day Relative Strength Index (RSI) to gauge overbought/oversold conditions
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI when not enough data
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        if features[0] < 0:  # If price change is negative, mild positive for SELL
            reward += 10.0  # Positive reward for selling during high risk
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += features[1] * 20.0  # Strong reward for MACD aligning with trend direction

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # RSI < 30 indicates oversold
            reward += 15.0  # Reward for considering buy
        elif features[3] > 70:  # RSI > 70 indicates overbought
            reward += 15.0  # Reward for considering sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))