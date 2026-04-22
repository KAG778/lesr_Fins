import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 50  # Neutral RSI
        deltas = np.diff(prices[-period:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices)
    features.append(rsi)

    # Feature 2: Average True Range (ATR) for volatility measurement
    def calculate_atr(prices, highs, lows, period=14):
        if len(prices) < period:
            return 0
        tr = np.maximum(highs[-period:] - lows[-period:], 
                        np.maximum(np.abs(highs[-period:] - prices[-period-1:-1]),
                                   np.abs(lows[-period:] - prices[-period-1:-1])))
        return np.mean(tr)

    high_prices = s[2::6]
    low_prices = s[3::6]
    atr = calculate_atr(closing_prices, high_prices, low_prices)
    features.append(atr)

    # Feature 3: Price Change Over Last 5 Days
    if len(closing_prices) >= 5:
        price_change = closing_prices[-1] - closing_prices[-5]
    else:
        price_change = 0
    features.append(price_change)

    # Feature 4: Volume Change Percentage
    if len(volumes) >= 2:
        volume_change_pct = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1e-10)  # Avoid division by zero
    else:
        volume_change_pct = 0
    features.append(volume_change_pct)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[0:120])  # Use the raw state for variability
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for BUY
        reward += 10 if enhanced_s[123] < 0 else 0  # Mild positive for SELL
    elif risk_level > 0.4 * historical_std:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold:
            reward += 30  # Strong positive for upward features
        elif trend_direction < -trend_threshold:
            reward += 30  # Strong positive for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += 20  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)