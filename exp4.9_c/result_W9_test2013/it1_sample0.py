import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Price Change Percentage (last 5 days)
    if len(closing_prices) >= 6:
        price_change_pct = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # Feature 2: Moving Average Convergence Divergence (MACD)
    short_ma = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else closing_prices[-1]
    long_ma = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else closing_prices[-1]
    macd = short_ma - long_ma
    features.append(macd)

    # Feature 3: Average True Range (ATR) to measure volatility
    if len(closing_prices) >= 14:
        true_ranges = np.maximum(closing_prices[1:] - closing_prices[:-1], np.maximum(closing_prices[1:] - closing_prices[:-1], closing_prices[:-1] - closing_prices[1:]))
        atr = np.mean(true_ranges[-14:])  # ATR over the last 14 days
    else:
        atr = 0
    features.append(atr)

    # Feature 4: Volume Weighted Average Price (VWAP)
    if len(volumes) >= 5 and len(closing_prices) >= 5:
        vwap = np.sum(closing_prices[-5:] * volumes[-5:]) / np.sum(volumes[-5:])
    else:
        vwap = closing_prices[-1] if len(closing_prices) > 0 else 0
    features.append(vwap)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract newly calculated features
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= 50 if features[0] > 0 else 20  # Using feature 0 (price change) to assess buy alignment
        # Mild positive reward for SELL-aligned features
        reward += 10 if features[0] < 0 else 0  # Assuming feature 0 indicates bearish signal
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= 25  # Adjusted to be more severe

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Aligning with an upward trend
            reward += 30  # Strong positive reward for bullish signals
        elif trend_direction < -0.3 and features[0] < 0:  # Aligning with a downward trend
            reward += 30  # Strong positive reward for bearish signals

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Assuming feature 0 indicates bearish signal
            reward += 15  # Reward for mean-reversion opportunities
        else:
            reward -= 5  # Penalize if buying in a sideways market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(reward, 100))

    return reward