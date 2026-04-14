import numpy as np

def revise_state(s):
    closing_prices = s[0::6]
    n = len(closing_prices)

    # Feature 1: Daily Returns (percentage change)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if n > 1 else np.array([0])
    daily_returns = np.concatenate(([0], daily_returns))  # Fill first element with 0 for shape compatibility

    # Feature 2: 14-day Relative Strength Index (RSI)
    window = 14
    if len(closing_prices) < window:
        rsi = np.zeros_like(closing_prices)
    else:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.convolve(gain, np.ones(window) / window, mode='valid')
        avg_loss = np.convolve(loss, np.ones(window) / window, mode='valid')
        rs = np.concatenate(([0]*window, avg_gain / (avg_loss + 1e-10)))
        rsi = 100 - (100 / (1 + rs))

    # Feature 3: 20-day Moving Average (MA)
    ma = np.convolve(closing_prices, np.ones(20) / 20, mode='valid')
    ma = np.concatenate(([0]*19, ma))  # Prepend zeros for the first '19' days

    # Feature 4: Historical Volatility (Last 20 days)
    if len(closing_prices) >= 20:
        historical_volatility = np.std(daily_returns[-20:])  # Based on last 20 daily returns
    else:
        historical_volatility = 0.0

    # Feature 5: Price Momentum (last price - moving average)
    price_momentum = closing_prices[-1] - ma[-1] if len(ma) > 0 else 0

    features = np.array([daily_returns[-1], rsi[-1], ma[-1], historical_volatility, price_momentum])
    return features

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    # Extract features
    daily_return = features[0]
    rsi = features[1]
    moving_average = features[2]
    historical_volatility = features[3]
    price_momentum = features[4]

    # Initialize reward
    reward = 0.0

    # Dynamic risk thresholds based on historical volatility
    high_risk_threshold = 0.7 * (1 + historical_volatility)
    moderate_risk_threshold = 0.4 * (1 + historical_volatility)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        if daily_return > 0:  # BUY-aligned feature
            reward = np.random.uniform(-50, -30)  # Strong negative for risky BUY
        else:  # SELL-aligned feature
            reward = np.random.uniform(5, 10)  # Mild positive for SELL
    elif risk_level > moderate_risk_threshold:
        if daily_return > 0:  # BUY signal
            reward = np.random.uniform(-20, -10)  # Moderate negative for risky BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < moderate_risk_threshold:
        if trend_direction > 0.3 and price_momentum > 0:  # Uptrend & positive momentum
            reward += np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < -0.3 and price_momentum < 0:  # Downtrend & negative momentum
            reward += np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold condition
            reward += np.random.uniform(5, 15)  # Positive reward for mean-reversion buying
        elif rsi > 70:  # Overbought condition
            reward += np.random.uniform(5, 15)  # Positive reward for mean-reversion selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < moderate_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range