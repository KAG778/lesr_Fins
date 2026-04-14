import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    features = []
    
    # Feature 1: Price Change Percentage over the last 5, 10, and 20 days
    change_5d = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) >= 6 else 0
    change_10d = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) >= 11 else 0
    change_20d = (closing_prices[-1] - closing_prices[-21]) / closing_prices[-21] if len(closing_prices) >= 21 else 0
    features.extend([change_5d, change_10d, change_20d])

    # Feature 2: Average True Range (ATR) for volatility
    true_ranges = np.maximum(0, np.maximum(closing_prices[1:] - closing_prices[:-1], 
                                   np.maximum(closing_prices[1:] - closing_prices[:-1], closing_prices[:-1] - closing_prices[1:])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    features.append(atr)

    # Feature 3: Moving Average of RSI (14 days)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.abs(np.where(deltas < 0, deltas, 0)).mean()
        rs = gains / losses if losses > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsi_ma = np.mean([rsi] + [np.mean(np.diff(closing_prices[i-14:i])) for i in range(14, len(closing_prices))])
                          if len(closing_prices) > 14 else [rsi])
    else:
        rsi_ma = 50  # Default if not enough data
    features.append(rsi_ma)

    # Feature 4: Volume Moving Average
    if len(volumes) >= 14:
        volume_ma = np.mean(volumes[-14:])
    else:
        volume_ma = volumes[-1] if len(volumes) > 0 else 0
    features.append(volume_ma)

    # Feature 5: Trend Strength (using linear regression slope)
    if len(closing_prices) >= 10:
        x = np.arange(10)
        y = closing_prices[-10:]
        slope = np.polyfit(x, y, 1)[0]  # Linear regression slope
    else:
        slope = 0
    features.append(slope)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        reward += np.random.uniform(5, 15)    # Mild positive reward for SELL
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += np.random.uniform(10, 30)  # Positive reward for alignment with trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion behavior

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))