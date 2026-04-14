import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extract closing prices from raw state
    features = []

    # 1. Rate of Change (ROC)
    try:
        roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]  # Change from last day to the day before
    except ZeroDivisionError:
        roc = 0.0
    features.append(roc)

    # 2. Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Mean gain
    loss = np.abs(np.where(delta < 0, delta, 0)).mean()  # Mean loss
    try:
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    except ZeroDivisionError:
        rsi = 50  # Neutral RSI
    features.append(rsi)

    # 3. Moving Average Convergence Divergence (MACD)
    short_term_ema = np.mean(closing_prices[-12:])  # Short-term EMA (last 12 days)
    long_term_ema = np.mean(closing_prices[-26:])  # Long-term EMA (last 26 days)
    macd = short_term_ema - long_term_ema
    features.append(macd)

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Negative for BUY actions
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)  # Slightly negative for BUY actions

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive for BUY signals
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive for SELL signals

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += np.random.uniform(5, 15)  # Positive for mean-reversion actions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clamp the reward between -100 and 100