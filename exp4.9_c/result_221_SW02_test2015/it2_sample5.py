import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes
    num_days = len(closing_prices)

    # Feature 1: Price Momentum (percentage change from the previous day)
    if num_days > 1:
        price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        price_momentum = 0
    features.append(price_momentum)

    # Feature 2: Relative Strength Index (RSI) for momentum and potential reversals
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)
    features.append(rsi)

    # Feature 3: Average True Range (ATR) for volatility measurement
    def compute_atr(highs, lows, closes, period=14):
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    high_prices = s[2::6]
    low_prices = s[3::6]
    atr = compute_atr(high_prices, low_prices, closing_prices)
    features.append(atr)

    # Feature 4: Volume Change (relative change)
    if len(volumes) >= 6:
        avg_volume_current = np.mean(volumes[-5:])
        avg_volume_previous = np.mean(volumes[-10:-5]) if len(volumes) > 10 else avg_volume_current
        volume_change = (avg_volume_current - avg_volume_previous) / avg_volume_previous if avg_volume_previous != 0 else 0
    else:
        volume_change = 0
    features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(features) if len(features) > 0 else 1  # Prevent division by zero

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(40, 60) * historical_std  # Strong negative for BUY
        reward += np.random.uniform(5, 15) * historical_std if features[0] < 0 else 0  # Mild positive for SELL
    elif risk_level > 0.4:
        reward -= 20 * historical_std  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and momentum confirms
            reward += 15 * historical_std  # Positive reward for following the trend
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and momentum confirms
            reward += 15 * historical_std  # Positive reward for following the trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Assuming RSI indicates oversold
            reward += 15 * historical_std  # Reward for buying in an oversold condition
        elif features[1] > 70:  # Assuming RSI indicates overbought
            reward -= 15 * historical_std  # Penalize for buying in an overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return float(np.clip(reward, -100, 100))