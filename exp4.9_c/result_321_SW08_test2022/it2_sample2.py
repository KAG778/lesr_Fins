import numpy as np

def revise_state(s):
    features = []

    # Reshape the raw state into a 20-day view with 6 features each
    days = s.reshape((20, 6))
    closing_prices = days[:, 0]  # Closing prices

    # Feature 1: Daily Return Volatility (Standard Deviation of Daily Returns)
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices
    daily_return_volatility = np.std(daily_returns)
    features.append(daily_return_volatility)

    # Feature 2: Average True Range (ATR) to measure volatility
    high_prices = days[:, 3]
    low_prices = days[:, 5]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:],
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    features.append(atr)

    # Feature 3: Rate of Change (Momentum Indicator)
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 6 else 0
    features.append(momentum)

    # Feature 4: Percentage Drawdown from the highest price in the last 20 days
    max_price = np.max(closing_prices)
    drawdown = (max_price - closing_prices[-1]) / max_price if max_price > 0 else 0
    features.append(drawdown)

    # Feature 5: RSI with historical thresholds to detect overbought/oversold conditions
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0  # Initialize reward

    # Calculate relative thresholds for risk management based on historical std
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -50  # Strong negative for high-risk BUY
        # Mild positive for SELL-aligned features
        if features[2] < 0:  # Assuming feature[2] is momentum
            reward += 10  # Mild positive reward for selling
    elif risk_level > risk_threshold_moderate:
        reward += -20  # Moderate negative for elevated risk BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0.3 and features[2] > 0:  # Uptrend and positive momentum
            reward += 20  # Positive reward for correct direction
        elif trend_direction < -0.3 and features[2] < 0:  # Downtrend and negative momentum
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_threshold_moderate:
        reward += 10  # Reward for mean-reversion actions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward