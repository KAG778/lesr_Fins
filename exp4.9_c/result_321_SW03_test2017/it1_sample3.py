import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    n = len(closing_prices)
    
    # Feature 1: Average True Range (ATR) for volatility
    if n < 2:
        atr = 0
    else:
        high_prices = s[2::6]
        low_prices = s[3::6]
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 0

    # Feature 2: Moving Average Convergence Divergence (MACD)
    if n < 26:
        macd = 0
    else:
        short_ema = np.mean(closing_prices[-12:])  # 12-day EMA
        long_ema = np.mean(closing_prices[-26:])   # 26-day EMA
        macd = short_ema - long_ema

    # Feature 3: Z-score of daily returns to adapt to market conditions
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    z_score = (np.mean(daily_returns) - np.mean(daily_returns[-14:])) / (np.std(daily_returns[-14:]) if np.std(daily_returns[-14:]) != 0 else 1)

    features = [atr, macd, z_score]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    atr = features[0]
    macd = features[1]
    z_score = features[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    historical_mean = 0.01  # Placeholders, should be replaced with historical data
    historical_std = 0.02   # Placeholders, should be replaced with historical data
    dynamic_high_risk = historical_mean + 2 * historical_std  # Dynamic risk threshold

    if risk_level > dynamic_high_risk:
        if z_score > 0:  # Assuming z_score > 0 indicates a BUY trend
            reward = -50  # Strong negative for risky BUY
        else:
            reward = 10  # Mild positive for SELL

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and macd > 0:  # Uptrend aligned with positive MACD
            reward += 20  # Positive reward for correct trend-following
        elif trend_direction < -0.3 and macd < 0:  # Downtrend aligned with negative MACD
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if z_score < -1:  # Oversold situation
            reward += 15  # Reward for mean-reversion buying
        elif z_score > 1:  # Overbought situation
            reward += 15  # Reward for mean-reversion selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range