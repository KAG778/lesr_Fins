import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]          # Trading volumes

    # Feature 1: Daily Return Volatility (standard deviation of daily returns over the last 20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    daily_return_volatility = np.std(daily_returns[-20:]) if len(daily_returns) > 20 else 0
    features.append(daily_return_volatility)

    # Feature 2: Mean Reversion Distance (current price relative to a moving average)
    moving_avg = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    mean_reversion_distance = (closing_prices[-1] - moving_avg) / moving_avg
    features.append(mean_reversion_distance)

    # Feature 3: Crisis Indicator (1 if recent volatility exceeds historical average, else 0)
    historical_volatility = np.std(daily_returns[-60:]) if len(daily_returns) >= 60 else 0  # Historical volatility over the last 60 days
    crisis_indicator = 1 if daily_return_volatility > historical_volatility else 0
    features.append(crisis_indicator)

    # Feature 4: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    features.append(price_momentum)

    # Feature 5: Average Trading Volume Change (percentage change in volume)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) > 20 else 0
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_returns = enhanced_s[123:]  # Assuming features start at index 123
    mean_return = np.mean(historical_returns)
    std_return = np.std(historical_returns)

    # Define relative thresholds
    high_risk_threshold = mean_return + 2 * std_return
    low_risk_threshold = mean_return + std_return

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 20 if trend_direction < 0 else 0  # Mild positive reward for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:
            reward += 15  # Positive reward for upward alignment
        else:
            reward += 15  # Positive reward for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features
        reward -= 5   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds