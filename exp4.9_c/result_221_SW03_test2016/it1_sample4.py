import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]
    volumes = s[4::6]
    
    # Feature 1: Average True Range (ATR) over the last 14 days
    # Calculate True Range
    highs = s[2::6]  # Extract highs
    lows = s[3::6]   # Extract lows
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closing_prices[:-1]), np.abs(lows[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.nan
    
    # Feature 2: Bollinger Bands (20-day moving average +/- 2 std dev)
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_avg + (2 * std_dev)
        lower_band = moving_avg - (2 * std_dev)
    else:
        upper_band = lower_band = np.nan
    
    # Feature 3: Exponential Moving Average (EMA) crossover
    if len(closing_prices) >= 26:
        ema_short = np.mean(closing_prices[-12:])  # 12-day EMA
        ema_long = np.mean(closing_prices[-26:])   # 26-day EMA
    else:
        ema_short = ema_long = np.nan
    
    features = [atr, upper_band, lower_band, ema_short - ema_long]  # Adding EMA difference for momentum
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    historical_std = np.std(enhanced_s[123:])  # Use the historical std of features for relative thresholds
    risk_thresholds = {
        'low': 0.4 * historical_std,
        'medium': 0.7 * historical_std,
        'high': 1.0 * historical_std
    }

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_thresholds['high']:
        reward -= 50  # Strong negative reward for BUY
        reward += 10   # Mild positive reward for SELL
    elif risk_level > risk_thresholds['medium']:
        reward -= 20  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_thresholds['medium']:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_thresholds['low']:
        rsi = enhanced_s[123]  # Assuming RSI is the first feature in revised state
        if rsi < 30:  # Oversold condition
            reward += 10  # Reward for potential buy
        elif rsi > 70:  # Overbought condition
            reward += 10  # Reward for potential sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds