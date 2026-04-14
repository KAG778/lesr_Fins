import numpy as np

def revise_state(s):
    # Extracting closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0 (closing prices)
    
    # Feature 1: 14-day Relative Strength Index (RSI)
    def compute_rsi(prices, window=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain) if len(gain) > 0 else 0
        avg_loss = np.mean(loss) if len(loss) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    # Feature 2: Moving Average Convergence Divergence (MACD)
    exp1 = np.mean(closing_prices[-12:])  # Short-term (12-day)
    exp2 = np.mean(closing_prices[-26:])  # Long-term (26-day)
    macd = exp1 - exp2

    # Feature 3: Bollinger Bands (mean and std deviation)
    rolling_mean = np.mean(closing_prices[-20:])
    rolling_std = np.std(closing_prices[-20:])
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    price_position = (closing_prices[-1] - lower_band) / (upper_band - lower_band)  # Normalized position

    # Feature 4: Crisis Detection (high volatility period)
    recent_volatility = np.std(np.diff(closing_prices[-20:]))  # 20-day rolling volatility
    crisis_indicator = 1 if recent_volatility > np.mean(np.std(np.diff(closing_prices[-100:]))) else 0

    features = [rsi, macd, price_position, crisis_indicator]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    rsi = features[0]
    macd = features[1]
    price_position = features[2]
    crisis_indicator = features[3]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40 if features[0] > 0 else 10  # Strong negative for risky BUY, mild positive for SELL
        return np.clip(reward, -100, 100)
    elif risk_level > 0.4:
        reward += -10 if features[0] > 0 else 0  # Moderate negative for risky BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and macd > 0:  # Uptrend & bullish signal
            reward += 20
        elif trend_direction < 0 and macd < 0:  # Downtrend & bearish signal
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if price_position < 0.2:  # Oversold condition
            reward += 15
        elif price_position > 0.8:  # Overbought condition
            reward += -15  # Penalize chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range