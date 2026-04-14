import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]          # Trading volumes

    # Feature 1: Average Daily Return Over Last 20 Days
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    avg_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(avg_daily_return)

    # Feature 2: Rolling Volatility (standard deviation of the last 20 days of returns)
    rolling_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(rolling_volatility)

    # Feature 3: Mean Reversion Distance (current price relative to a moving average)
    moving_avg = np.mean(closing_prices[-20:])  # 20-day moving average
    mean_reversion_distance = (closing_prices[-1] - moving_avg) / moving_avg
    features.append(mean_reversion_distance)

    # Feature 4: Crisis Indicator (1 if recent volatility exceeds historical threshold, else 0)
    historical_volatility = np.std(closing_prices[-60:])  # Historical volatility over 60 days
    crisis_indicator = 1 if rolling_volatility > historical_volatility else 0
    features.append(crisis_indicator)

    # Feature 5: Average Trading Volume Change (percentage change)
    avg_volume_change = np.mean(np.diff(volumes) / volumes[:-1]) if len(volumes) > 1 else 0
    features.append(avg_volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_returns = enhanced_s[123:]
    mean_return = np.mean(historical_returns)
    std_return = np.std(historical_returns)

    # Define dynamic risk thresholds
    high_risk_threshold = mean_return + 2 * std_return
    low_risk_threshold = mean_return - 2 * std_return

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # STRONG NEGATIVE reward for BUY-aligned features
        # MILD POSITIVE reward for SELL-aligned features (optional)
        reward += 10 if trend_direction < 0 else 0
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:
            reward += 15  # Positive reward for upward signals
        else:
            reward += 15  # Positive reward for downward signals

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds