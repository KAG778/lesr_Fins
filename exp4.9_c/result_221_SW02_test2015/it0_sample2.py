import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    closing_prices = s[0:120:6]  # Extract closing prices from the raw state
    days = 20  # Number of days in the state

    # 1. Calculate RSI
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    # 2. Calculate MACD
    def compute_macd(prices, short_window=12, long_window=26, signal_window=9):
        short_ema = np.mean(prices[-short_window:]) if len(prices) >= short_window else 0
        long_ema = np.mean(prices[-long_window:]) if len(prices) >= long_window else 0
        macd = short_ema - long_ema
        signal_line = np.mean(np.array([macd] + [0] * (signal_window - 1)))  # Simplified
        return macd - signal_line

    macd = compute_macd(closing_prices)

    # 3. Calculate ATR
    def compute_atr(prices, period=14):
        high_prices = s[2:120:6]  # Extract high prices
        low_prices = s[3:120:6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                   abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else 0
        return atr

    atr = compute_atr(closing_prices)

    features = [rsi, macd, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        return reward
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        features = enhanced_s[123:]
        if trend_direction > 0.3 and features[1] > 0:  # Assuming features[1] indicates upward features
            reward += 15  # Positive reward for upward features in uptrend
        elif trend_direction < -0.3 and features[1] < 0:  # Assuming features[1] indicates downward features
            reward += 15  # Positive reward for downward features in downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        features = enhanced_s[123:]
        if features[0] < 30:  # Assuming features[0] is RSI indicating oversold
            reward += 10  # Reward for buying in oversold
        elif features[0] > 70:  # Assuming features[0] is RSI indicating overbought
            reward += 10  # Reward for selling in overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward