import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]  # Extracting trading volumes
    num_days = len(closing_prices)

    # Feature 1: Relative Strength Index (RSI) to measure momentum and potential reversals
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    # Feature 2: Average True Range (ATR) for volatility measurement
    def compute_atr(highs, lows, closes, period=14):
        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(abs(highs[1:] - closes[:-1]), 
                                   abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else 0
        return atr

    high_prices = s[2::6]
    low_prices = s[3::6]
    atr = compute_atr(high_prices, low_prices, closing_prices)

    # Feature 3: Bollinger Bands for assessing volatility and overbought/oversold conditions
    if num_days >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        current_price = closing_prices[-1]
        bb_signal = 0
        if current_price > upper_band:
            bb_signal = 1  # Overbought
        elif current_price < lower_band:
            bb_signal = -1  # Oversold
    else:
        bb_signal = 0  # Neutral if not enough data

    # Append features to the list
    features.append(rsi)
    features.append(atr)
    features.append(bb_signal)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features from the enhanced state
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(40, 60)  # Strong negative for BUY
        if features[0] < 0:  # Assuming feature[0] indicates momentum suggesting SELL
            reward += np.random.uniform(5, 15)  # Mild positive for SELL
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 50:  # RSI indicating upward momentum
            reward += 15  # Positive reward for bullish alignment
        elif trend_direction < -0.3 and features[0] < 50:  # RSI indicating downward momentum
            reward += 15  # Positive reward for bearish alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] == -1:  # Assuming -1 indicates oversold
            reward += 15  # Reward for buying in an oversold condition
        elif features[2] == 1:  # Assuming 1 indicates overbought
            reward -= 15  # Penalize for buying in an overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds