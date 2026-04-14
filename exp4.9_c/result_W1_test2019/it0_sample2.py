import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    days = len(closing_prices)

    # Feature 1: Price Change Percentage
    price_change_pct = np.zeros(days - 1)
    for i in range(1, days):
        if closing_prices[i - 1] != 0:
            price_change_pct[i - 1] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]

    # Feature 2: MACD
    short_window = 12
    long_window = 26
    signal_window = 9

    exp1 = np.zeros(days)
    exp2 = np.zeros(days)
    macd = np.zeros(days)

    # Exponential Moving Average (EMA) calculation
    for i in range(days):
        if i < short_window:
            exp1[i] = np.nan
        else:
            exp1[i] = np.mean(closing_prices[i-short_window:i])
    
    for i in range(days):
        if i < long_window:
            exp2[i] = np.nan
        else:
            exp2[i] = np.mean(closing_prices[i-long_window:i])

    macd = exp1 - exp2

    # Feature 3: RSI Calculation
    gains = np.where(price_change_pct > 0, price_change_pct, 0)
    losses = np.where(price_change_pct < 0, -price_change_pct, 0)

    avg_gain = np.zeros(days)
    avg_loss = np.zeros(days)
    
    # Calculate average gains and losses
    avg_gain[0] = np.mean(gains[:14]) if np.any(gains[:14]) else 0
    avg_loss[0] = np.mean(losses[:14]) if np.any(losses[:14]) else 0

    for i in range(1, days):
        avg_gain[i] = (avg_gain[i-1] * 13 + gains[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + losses[i-1]) / 14

    rs = avg_gain / avg_loss
    rs[np.isnan(rs)] = 0  # Handle division by zero

    rsi = 100 - (100 / (1 + rs))

    # Collect features
    features = [price_change_pct[-1], macd[-1], rsi[-1]]  # Use the most recent values
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        return reward  # Early exit due to high risk
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20  # Reward for positive trend
        else:
            reward += 20  # Reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward