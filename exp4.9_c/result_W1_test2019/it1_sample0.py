import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element)
    days = len(closing_prices)

    # Feature 1: Recent Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Exponential Moving Average (EMA) - 10 days
    ema_short = np.zeros(days)
    multiplier = 2 / (10 + 1)
    ema_short[0] = closing_prices[0]  # Start with the first price
    for i in range(1, days):
        ema_short[i] = (closing_prices[i] - ema_short[i - 1]) * multiplier + ema_short[i - 1]

    # Feature 3: Average True Range (ATR) - 14 days for volatility measurement
    true_ranges = np.maximum(
        closing_prices[1:] - closing_prices[:-1],
        np.maximum(
            np.abs(closing_prices[1:] - closing_prices[:-1]),
            np.abs(closing_prices[1:] - closing_prices[:-1])
        )
    )
    atr = np.zeros(days)
    for i in range(len(true_ranges)):
        if i < 13:
            atr[i + 1] = np.nan
        else:
            atr[i + 1] = np.mean(true_ranges[i - 13:i + 1])

    # Feature 4: Volume Change Percentage - to identify volume spikes
    volume_change_pct = (s[4:120:6][-1] - s[4:120:6][-2]) / s[4:120:6][-2] if s[4:120:6][-2] != 0 else 0

    # Collect features
    features = [price_change_pct, ema_short[-1], atr[-1], volume_change_pct]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for relative thresholds
    # Assuming that the values of past risk levels, trend directions, and volatility levels are available
    # Here we use dummy values; replace them with actual historical data
    historical_std_risk = 0.2  # Example value
    historical_std_trend = 0.3  # Example value
    historical_std_volatility = 0.2  # Example value

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_std_risk + 0.5:  # Strong negative for BUY
        reward -= 50
    elif risk_level > historical_std_risk:  # Mild positive for SELL
        reward += 10

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > historical_std_trend and risk_level < historical_std_risk:
        reward += 20 * np.sign(trend_direction)  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < historical_std_trend and risk_level < historical_std_risk:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std_volatility + 0.5 and risk_level < historical_std_risk:
        reward *= 0.5  # Reduce reward magnitude

    # Clip the reward to ensure it remains within [-100, 100]
    return np.clip(reward, -100, 100)