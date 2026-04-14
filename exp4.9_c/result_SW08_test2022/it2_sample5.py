import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: Daily Returns (percentage changes)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    daily_returns = np.concatenate(([0], daily_returns))  # Align lengths
    features.append(np.mean(daily_returns))  # Average daily return

    # Feature 2: Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
    features.append(volatility)

    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss > 0 else 100
    features.append(rsi)

    # Feature 4: Crisis Indicator (number of days with returns below a threshold)
    crisis_threshold = np.mean(daily_returns) - 2 * np.std(daily_returns) if len(daily_returns) > 1 else 0
    crisis_days = np.sum(daily_returns < crisis_threshold)
    features.append(crisis_days)

    # Feature 5: Price Momentum over the last 5 days
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0
    features.append(price_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds based on the features
    historical_returns = enhanced_s[123:]
    mean_return = np.mean(historical_returns)
    std_return = np.std(historical_returns)

    # Dynamic thresholds
    high_risk_threshold = mean_return + 2 * std_return
    low_risk_threshold = mean_return - 2 * std_return

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10 if trend_direction < 0 else 0  # Mild positive reward for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:  # Uptrend
            reward += 15  # Positive reward for upward momentum
        else:  # Downtrend
            reward += 15  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds