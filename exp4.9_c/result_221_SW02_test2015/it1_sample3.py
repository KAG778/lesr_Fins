import numpy as np

def revise_state(s):
    features = []
    
    # Extract relevant data from the state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    
    # 1. Calculate Price Momentum (percentage change)
    if len(closing_prices) >= 2:
        price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        price_momentum = 0
    features.append(price_momentum)
    
    # 2. Average True Range (ATR) for volatility
    def compute_atr(highs, lows, closes, period=14):
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    atr = compute_atr(high_prices, low_prices, closing_prices)
    features.append(atr)
    
    # 3. Relative Volume Change (current volume vs historical average)
    if len(volumes) >= 10:
        avg_volume = np.mean(volumes[-10:])
        current_volume = volumes[-1]
        volume_change = (current_volume - avg_volume) / avg_volume if avg_volume != 0 else 0
    else:
        volume_change = 0
    features.append(volume_change)
    
    # 4. RSI (Relative Strength Index) for momentum
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices) if len(closing_prices) >= 14 else 50  # Default to neutral if not enough data
    features.append(rsi)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract the new features
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for risky BUY
        if features[0] < 0:  # If momentum suggests a SELL
            reward += 10  # Mild positive reward for SELL
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for risky BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if (trend_direction > 0 and features[0] > 0) or (trend_direction < 0 and features[0] < 0):
            reward += 15  # Positive reward for aligned momentum
        if features[3] > 70 and trend_direction > 0:
            reward -= 10  # Penalize overbought condition in an uptrend
        elif features[3] < 30 and trend_direction < 0:
            reward -= 10  # Penalize oversold condition in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # Assuming RSI indicates oversold
            reward += 15  # Reward for buying in oversold condition
        elif features[3] > 70:  # Assuming RSI indicates overbought
            reward -= 15  # Penalize for buying in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds