import numpy as np

def revise_state(s):
    features = []
    
    # Extract relevant data from the state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices

    num_days = len(closing_prices)

    # Feature 1: Price Momentum (percentage change)
    if num_days >= 2:
        price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        price_momentum = 0
    features.append(price_momentum)

    # Feature 2: Average True Range (ATR) for volatility
    def compute_atr(highs, lows, closes, period=14):
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    atr = compute_atr(high_prices, low_prices, closing_prices)
    features.append(atr)

    # Feature 3: Relative Volume Change (current volume vs historical average)
    if len(volumes) >= 10:
        avg_volume = np.mean(volumes[-10:])
        current_volume = volumes[-1]
        volume_change = (current_volume - avg_volume) / avg_volume if avg_volume != 0 else 0
    else:
        volume_change = 0
    features.append(volume_change)

    # Feature 4: Relative Strength Index (RSI) for momentum
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices) if len(closing_prices) >= 14 else 50  # Default to neutral if insufficient data
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extracting features
    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_std = np.std(features) if len(features) > 0 else 1  # Prevent division by zero

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY
        if features[0] < 0:  # Assuming feature[0] indicates momentum suggesting SELL
            reward += np.random.uniform(5, 15)  # MILD POSITIVE for SELL
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += 15  # Positive reward for aligning with the trend
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += 15  # Positive reward for aligning with the trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # RSI indicating oversold condition
            reward += 15  # Reward for buying in an oversold condition
        elif features[3] > 70:  # RSI indicating overbought condition
            reward -= 15  # Penalize for buying in an overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds