import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: Price Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(roc)

    # Feature 2: Average True Range (ATR) for volatility
    high_prices = s[2::6]  # High prices
    low_prices = s[3::6]   # Low prices
    tr = np.maximum(high_prices[-1] - low_prices[-1], 
                    np.maximum(np.abs(high_prices[-1] - closing_prices[-2]), 
                               np.abs(low_prices[-1] - closing_prices[-2])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else tr[-1]  # 14-period ATR
    features.append(atr)

    # Feature 3: Updated RSI with dynamic thresholds
    delta = np.diff(closing_prices)
    gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Normalize RSI thresholds based on historical data
    mean_rsi = np.mean(rsi)  # Mean RSI
    std_rsi = np.std(rsi)  # Standard deviation of RSI
    normalized_rsi = (rsi - mean_rsi) / std_rsi if std_rsi != 0 else rsi  # Z-score normalization
    features.append(normalized_rsi)

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
        reward += -50  # STRONG NEGATIVE reward for BUY-aligned features
        # MILD POSITIVE reward for SELL-aligned features
        reward += 10
        return np.clip(reward, -100, 100)
    
    if risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        momentum = features[0]  # ROC as momentum indicator
        if trend_direction > 0.3 and momentum > 0:  # Uptrend with positive momentum
            reward += 20
        elif trend_direction < -0.3 and momentum < 0:  # Downtrend with negative momentum
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        rsi = features[2]  # Normalized RSI
        if rsi < -1:  # Oversold condition
            reward += 15  # Encourage buying
        elif rsi > 1:  # Overbought condition
            reward += 15  # Encourage selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward stays within bounds