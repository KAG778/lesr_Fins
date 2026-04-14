import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    highs = s[2:120:6]            # Extract high prices
    lows = s[3:120:6]             # Extract low prices

    # Feature 1: Price Momentum (current closing price vs previous closing price)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / (closing_prices[-2] + 1e-10)

    # Feature 2: Relative Strength Index (RSI) over the last 14 days
    delta = np.diff(closing_prices)
    gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs)) if (gain + loss) > 0 else 0

    # Feature 3: Average Directional Index (ADX) to measure trend strength
    if len(closing_prices) >= 14:
        up_moves = np.maximum(0, highs[1:] - highs[:-1])
        down_moves = np.maximum(0, lows[:-1] - lows[1:])
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closing_prices[:-1]), np.abs(lows[1:] - closing_prices[:-1])))
        pdi = (np.mean(up_moves) / np.mean(tr)) * 100 if np.mean(tr) > 0 else 0
        mdi = (np.mean(down_moves) / np.mean(tr)) * 100 if np.mean(tr) > 0 else 0
        adx = np.abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100 if (pdi + mdi) > 0 else 0
    else:
        adx = 0

    # Feature 4: Volume Weighted Average Price (VWAP) Position
    total_volume = np.sum(volumes)
    if total_volume > 0:
        vwap = np.sum(closing_prices * volumes) / total_volume
        vwap_position = (closing_prices[-1] - vwap) / (vwap + 1e-10)
    else:
        vwap_position = 0.0

    return np.array([price_momentum, rsi, adx, vwap_position])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50.0 * (features[0] if features[0] > 0 else 1)  # Strong negative for BUY-aligned features
        reward += 10.0 * (features[3] < 0)  # Mild positive for SELL-aligned features based on VWAP position
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 20.0  # Reward for following trend based on price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # RSI below 30 indicates oversold condition
            reward += 10.0  # Encourage BUY
        elif features[1] > 70:  # RSI above 70 indicates overbought condition
            reward += 10.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))