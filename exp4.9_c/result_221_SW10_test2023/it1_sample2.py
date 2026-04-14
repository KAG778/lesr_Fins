import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and calculate returns
    closing_prices = s[0:120:6]
    if len(closing_prices) > 1:
        daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    else:
        daily_returns = np.array([0])
    
    # Feature 1: Mean Return
    mean_return = np.mean(daily_returns)
    features.append(mean_return)

    # Feature 2: Volatility (Standard Deviation of Returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    # Feature 3: Price Momentum (current close - close n days ago)
    n_days = 5
    if len(closing_prices) > n_days:
        momentum = (closing_prices[-1] - closing_prices[-n_days]) / closing_prices[-n_days]
    else:
        momentum = 0
    features.append(momentum)

    # Feature 4: Exponential Moving Average (EMA) for trend detection
    if len(closing_prices) >= 12:
        ema_short = np.mean(closing_prices[-12:])  # Short-term EMA
        ema_long = np.mean(closing_prices[-26:])   # Long-term EMA
        ema_signal = ema_short - ema_long
    else:
        ema_signal = 0
    features.append(ema_signal)

    # Feature 5: Average True Range (ATR) for volatility measure
    highs = s[2:120:6]
    lows = s[3:120:6]
    if len(highs) > 1 and len(lows) > 1:
        true_ranges = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
        atr = np.mean(true_ranges[-14:])  # 14-day ATR
    else:
        atr = 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    volatility_threshold = np.mean(features[1]) + 2 * np.std(features[1])  # Using feature 1 as volatility measure
    momentum_threshold = np.mean(features[2])  # Using feature 2 as momentum measure

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[2] > momentum_threshold:  # Positive momentum
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        else:
            reward += np.random.uniform(5, 10)

    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[2] > momentum_threshold:
            reward -= 10

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[2] > 0:  # Uptrend and positive momentum
            reward += np.random.uniform(10, 20)
        elif trend_direction < 0 and features[2] < 0:  # Downtrend and negative momentum
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0:  # Oversold condition
            reward += np.random.uniform(5, 15)
        elif features[2] > 0:  # Overbought condition
            reward -= np.random.uniform(5, 15)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within the bounds
    return float(np.clip(reward, -100, 100))