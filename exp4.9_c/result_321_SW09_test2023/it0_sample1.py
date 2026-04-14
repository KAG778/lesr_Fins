import numpy as np

def revise_state(s):
    # s: 120d raw state (OHLCV data)
    closing_prices = s[::6]  # Closing prices (indices 0, 6, 12, ..., 114)
    high_prices = s[::6 + 2]  # High prices (indices 2, 8, 14, ..., 116)
    low_prices = s[::6 + 3]  # Low prices (indices 3, 9, 15, ..., 117)
    
    # Feature 1: 14-day Relative Strength Index (RSI)
    # Using min(max(0, value), 100) to clamp the RSI between 0 and 100
    price_changes = np.diff(closing_prices)
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    
    average_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    average_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    
    rs = average_gain / average_loss if average_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 2: Moving Average Convergence Divergence (MACD)
    short_window = 12
    long_window = 26
    signal_window = 9
    
    short_ema = np.mean(closing_prices[-short_window:]) if len(closing_prices) >= short_window else 0
    long_ema = np.mean(closing_prices[-long_window:]) if len(closing_prices) >= long_window else 0
    macd = short_ema - long_ema

    # Feature 3: Price Action (Close - Open) as a momentum indicator
    price_action = closing_prices[-1] - s[-1]  # Last closing price - last opening price
    
    # Return the computed features
    features = [rsi, macd, price_action]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 70 and trend_direction > 0:  # RSI indicates overbought in an uptrend
            reward += 10.0  # Positive reward for confirming uptrend
        elif features[0] < 30 and trend_direction < 0:  # RSI indicates oversold in a downtrend
            reward += 10.0  # Positive reward for confirming downtrend
        reward += trend_direction * features[1] * 5.0  # MACD signal strength

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 30:  # Oversold condition
            reward += 15.0  # Mild positive reward for buying in oversold
        elif features[0] > 70:  # Overbought condition
            reward += 15.0  # Mild positive reward for selling in overbought

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))