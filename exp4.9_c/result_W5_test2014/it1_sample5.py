import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Volumes

    # Feature 1: Price Momentum (current close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0
    
    # Feature 2: Average True Range (ATR)
    def calculate_atr(prices, period=14):
        high_prices = s[1::6]
        low_prices = s[2::6]
        tr = np.maximum(high_prices[-1] - low_prices[-1], 
                        np.abs(high_prices[-1] - closing_prices[-2]), 
                        np.abs(low_prices[-1] - closing_prices[-2]))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    atr = calculate_atr(s)

    # Feature 3: Bollinger Bands
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    upper_band = moving_average + (2 * std_dev)
    lower_band = moving_average - (2 * std_dev)

    # Feature 4: RSI (Relative Strength Index)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))
    
    rsi = calculate_rsi(closing_prices)

    features = [price_momentum, atr, upper_band, lower_band, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]

    # Setting initial reward
    reward = 0.0

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40 if features[0] > 0 else 5  # Assuming feature[0] relates to BUY signal
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20 if features[0] > 0 else 0  # Assuming feature[0] relates to BUY signal

    # **Priority 2 — TREND FOLLOWING**
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Assuming feature[0] relates to upward signal
            reward += 10
        elif trend_direction < -0.3 and features[0] < 0:  # Assuming feature[0] relates to downward signal
            reward += 10

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[4] < 30:  # Assuming feature[4] relates to RSI (oversold)
            reward += 10  # Reward buying in oversold conditions
        elif features[4] > 70:  # Assuming feature[4] relates to RSI (overbought)
            reward += 10  # Reward selling in overbought conditions

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds