import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]          # Trading volumes

    # Feature 1: Daily Return (percentage change)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    daily_returns = np.concatenate(([0], daily_returns))  # Align lengths
    features.append(np.mean(daily_returns[-20:]))  # Average daily return over the last 20 days

    # Feature 2: Rolling Volatility (standard deviation of the last 20 days of returns)
    rolling_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(rolling_volatility)

    # Feature 3: Mean Reversion Indicator (current price relative to a 20-day moving average)
    moving_avg = np.mean(closing_prices[-20:])  # 20-day moving average
    mean_reversion_distance = (closing_prices[-1] - moving_avg) / moving_avg if moving_avg != 0 else 0
    features.append(mean_reversion_distance)

    # Feature 4: Volume Change (percentage change in volume)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_change)

    # Feature 5: Crisis Indicator (1 if recent volatility exceeds historical threshold, else 0)
    historical_volatility = np.std(daily_returns[-60:]) if len(daily_returns) >= 60 else 0  # Historical volatility over 60 days
    crisis_indicator = 1 if rolling_volatility > historical_volatility else 0
    features.append(crisis_indicator)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_returns = enhanced_s[123:]  # Assuming features start at index 123
    mean_return = np.mean(historical_returns) if len(historical_returns) > 0 else 0
    std_return = np.std(historical_returns) if len(historical_returns) > 0 else 0

    # Define dynamic thresholds based on historical data
    high_risk_threshold = mean_return + 2 * std_return
    low_risk_threshold = mean_return - 2 * std_return

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # STRONG NEGATIVE reward for BUY-aligned features
        reward += 10 if trend_direction < 0 else 0  # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate penalty for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 15 if trend_direction > 0 else -15  # Positive reward for upward momentum, negative for downward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds