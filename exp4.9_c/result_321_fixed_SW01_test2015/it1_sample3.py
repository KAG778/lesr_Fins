import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    num_days = len(closing_prices)

    # Feature 1: Price Momentum (current close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2] if num_days > 1 else 0.0

    # Feature 2: Standard Deviation of Last 14 Days (Volatility)
    price_volatility = np.std(closing_prices[-14:]) if num_days > 14 else 0.0

    # Feature 3: Average True Range (ATR) for volatility measure
    highs = s[3:120:6]  # Extract high prices
    lows = s[2:120:6]   # Extract low prices
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0.0

    # Feature 4: Volume Change (percentage change)
    volumes = s[4:120:6]  # Extract volumes
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 if num_days > 1 and volumes[-2] > 0 else 0.0

    # Feature 5: 14-Day RSI
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0]) if len(deltas[deltas > 0]) > 0 else 0
        loss = -np.mean(deltas[deltas < 0]) if len(deltas[deltas < 0]) > 0 else 0
        rs = gain / loss if loss > 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices) if num_days > 14 else 0.0

    features = [price_momentum, price_volatility, atr, volume_change, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY in high-risk conditions
        reward += 5.0 * abs(features[0])  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Align reward with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[4] < 30:  # RSI < 30: Oversold condition
            reward += 10.0  # Positive for buying in oversold condition
        elif features[4] > 70:  # RSI > 70: Overbought condition
            reward -= 10.0  # Negative for selling in overbought condition

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))