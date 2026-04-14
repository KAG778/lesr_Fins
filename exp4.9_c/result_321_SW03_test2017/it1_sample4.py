import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Every 6th element starting from index 0 (closing prices)
    
    # Feature 1: Daily Returns (percentage change)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    daily_returns = np.concatenate(([0], daily_returns))  # Fill first element with 0 for shape compatibility
    
    # Feature 2: 14-day Relative Strength Index (RSI)
    window_rsi = 14
    if len(closing_prices) < window_rsi:
        rsi = np.zeros_like(closing_prices)
    else:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.convolve(gain, np.ones(window_rsi)/window_rsi, mode='valid')
        avg_loss = np.convolve(loss, np.ones(window_rsi)/window_rsi, mode='valid')
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = np.concatenate(([0]*window_rsi, rsi))  # Prepend zeros

    # Feature 3: 20-day Moving Average (MA)
    ma_length = 20
    ma = np.convolve(closing_prices, np.ones(ma_length)/ma_length, mode='valid')
    ma = np.concatenate(([0]*(ma_length-1), ma))  # Prepend zeros

    # Feature 4: Crisis Indicator (detecting extreme drops)
    price_change = np.diff(closing_prices)
    crisis_indicator = np.where(price_change < -0.05, 1, 0)  # 5% drop

    # Feature 5: Trend Strength (momentum over last 5 days)
    trend_strength = np.sum(price_change[-5:]) if len(price_change) >= 5 else 0

    # Return computed features
    features = np.array([daily_returns[-1], rsi[-1], ma[-1], np.mean(crisis_indicator[-5:]), trend_strength])
    return features

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    daily_return = features[0]
    rsi = features[1]
    moving_average = features[2]
    crisis_indicator = features[3]
    trend_strength = features[4]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    # Use std of daily returns for relative thresholds
    historical_std = np.std(features[:20])  # Taking std of the last 20 daily returns for threshold
    if risk_level > 0.7:
        if daily_return > 0:  # BUY-aligned feature
            reward = np.random.uniform(-100, -50)  # Strong negative for risky BUY
        else:
            reward = np.random.uniform(10, 20)  # Mild positive for SELL
    elif risk_level > 0.4:
        if daily_return > 0:  # BUY signal
            reward = np.random.uniform(-30, -10)  # Moderate negative for risky BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and daily_return > 0:  # Uptrend & positive return
            reward += np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < -0.3 and daily_return < 0:  # Downtrend & negative return
            reward += np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if daily_return < 0:  # Oversold situation
            reward += np.random.uniform(5, 15)  # Positive reward for buying
        elif daily_return > 0:  # Overbought situation
            reward += np.random.uniform(-15, -5)  # Negative reward for buying

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range