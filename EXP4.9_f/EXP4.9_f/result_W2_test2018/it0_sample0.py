import numpy as np

def revise_state(s):
    # s: 120d raw state, containing 20 days of OHLCV data
    closing_prices = s[0::6]  # Extract closing prices, every 6th element starting from index 0
    days = len(closing_prices)  # Should be 20

    # Feature 1: Momentum (current closing price - closing price 5 days ago)
    momentum = closing_prices[0] - closing_prices[5] if days > 5 else 0

    # Feature 2: Historical Volatility (standard deviation of closing prices over the past 20 days)
    volatility = np.std(closing_prices) if days > 1 else 0

    # Feature 3: Moving Average Convergence (short-term vs long-term moving average)
    short_window = 5
    long_window = 20
    short_ma = np.mean(closing_prices[:short_window]) if days >= short_window else 0
    long_ma = np.mean(closing_prices) if days >= long_window else 0
    ma_convergence = short_ma - long_ma

    # Return the computed features
    features = [momentum, volatility, ma_convergence]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assuming feature[0] is aligned with BUY
            reward = np.random.uniform(-50, -30)  # STRONG NEGATIVE for BUY-aligned features
        else:
            reward = np.random.uniform(5, 10)  # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        if features[0] > 0:  # Assuming feature[0] is aligned with BUY
            reward = np.random.uniform(-20, -10)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # BUY aligned with uptrend
            reward = np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < 0 and features[0] < 0:  # SELL aligned with downtrend
            reward = np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Assuming feature[0] is aligned with BUY
            reward = np.random.uniform(5, 15)  # Reward mean-reversion features
        else:
            reward = np.random.uniform(-10, 0)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)