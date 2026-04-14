import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6][:20]  # Extract closing prices for the last 20 days
    volumes = s[4::6][:20]  # Extract trading volumes

    # Feature 1: Exponential Moving Average (EMA) for trend detection
    ema_span = 14
    ema = np.zeros_like(closing_prices)
    ema[:ema_span] = np.nan  # Set initial values to NaN
    ema[ema_span - 1] = np.mean(closing_prices[:ema_span])  # Calculate first EMA
    for i in range(ema_span, len(closing_prices)):
        ema[i] = (closing_prices[i] - ema[i - 1]) * (2 / (ema_span + 1)) + ema[i - 1]

    # Feature 2: Average True Range (ATR) for volatility measure
    high_prices = s[2::6][:20]
    low_prices = s[3::6][:20]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1][1:]), 
                               np.abs(low_prices[1:] - closing_prices[:-1][1:])))
    atr = np.mean(tr) if len(tr) > 0 else 0

    # Feature 3: Z-Score of daily returns for mean reversion
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    z_score = (daily_returns[-1] - np.mean(daily_returns)) / np.std(daily_returns) if len(daily_returns) > 1 else 0

    features = [ema[-1], atr, z_score]  # Include latest EMA, ATR and Z-Score
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate relative thresholds based on historical std
    historical_std = np.std(enhanced_s[123:]) if len(enhanced_s[123:]) > 0 else 1
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)   # Mild positive reward for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < low_risk_threshold:
        if trend_direction > 0:
            reward += np.random.uniform(10, 20)  # Reward for upward momentum
        else:
            reward += np.random.uniform(10, 20)  # Reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is capped within [-100, 100]
    return np.clip(reward, -100, 100)