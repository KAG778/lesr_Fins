import numpy as np

def revise_state(s):
    features = []

    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices

    # Feature 1: Average True Range (ATR) over the last 14 days
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0.0
    features.append(atr)

    # Feature 2: Exponential Moving Average (EMA) difference (10-day - 50-day)
    ema_short = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0.0
    ema_long = np.mean(closing_prices[-50:]) if len(closing_prices) >= 50 else 0.0
    features.append(ema_short - ema_long)

    # Feature 3: Z-score of recent returns for mean-reversion signal
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    mean_return = np.mean(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    std_return = np.std(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score)

    # Feature 4: Crisis Indicator (percentage drop from recent peak)
    recent_peak = np.max(closing_prices[-20:])  # Look at the last 20 days for peak
    crisis_indicator = (recent_peak - closing_prices[-1]) / recent_peak if recent_peak != 0 else 0.0
    features.append(crisis_indicator)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds for risk management based on historical data
    historical_std = np.std(enhanced_s[123:])  # Assuming features are already added
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward += -np.random.uniform(30, 50)  # Strong negative for BUY
        reward += np.random.uniform(10, 20)  # Mild positive for SELL
    elif risk_level > risk_threshold_moderate:
        reward += -np.random.uniform(5, 15)  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for upward trend
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the range [-100, 100]
    return np.clip(reward, -100, 100)