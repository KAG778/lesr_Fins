import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    
    # Extract closing prices and volumes from raw state
    closing_prices = s[0:120:6]  # every 6th element starting from index 0
    volumes = s[4:120:6]          # every 6th element starting from index 4 (volumes)

    features = []

    # 1. Price Momentum (current close - average close of the last 5 days)
    if len(closing_prices) > 5:
        momentum = closing_prices[0] - np.mean(closing_prices[1:6])
    else:
        momentum = 0
    features.append(momentum)

    # 2. Volume Weighted Average Price (VWAP)
    if len(volumes) > 5:
        vwap = np.sum(closing_prices[:5] * volumes[:5]) / np.sum(volumes[:5])
    else:
        vwap = closing_prices[0] if volumes[0] > 0 else 0
    features.append(vwap)

    # 3. Average True Range (ATR) for volatility measurement
    true_ranges = []
    for i in range(1, len(closing_prices)):
        high = s[2 + i * 6]  # high_prices
        low = s[3 + i * 6]   # low_prices
        close_prev = closing_prices[i-1]
        true_ranges.append(max(high - low, abs(high - close_prev), abs(low - close_prev)))
    
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) > 14 else np.std(closing_prices)  # ATR over 14 days
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate historical thresholds for risk and volatility
    historical_risk_threshold = np.mean([0.4, 0.6])  # Example threshold for risk level
    historical_volatility_threshold = np.mean([0.3, 0.5])  # Example threshold for volatility level

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY
        reward += np.random.uniform(0, 10)    # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < historical_risk_threshold:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 30)  # Reward for aligning with the trend
        elif trend_direction < 0:  # Downtrend
            reward -= np.random.uniform(10, 30)  # Penalty for opposing the trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds