import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    days = len(closing_prices)

    # Feature 1: Rate of Change (ROC) - measures the percentage change in price over a specified time period
    roc_period = 12  # Lookback period for ROC
    roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period] if days >= roc_period else 0

    # Feature 2: Bollinger Band Width - measures volatility via the width of the Bollinger Bands
    window = 20
    if days >= window:
        rolling_mean = np.mean(closing_prices[-window:])
        rolling_std = np.std(closing_prices[-window:])
        bb_width = (rolling_std / rolling_mean) if rolling_mean != 0 else 0
    else:
        bb_width = 0

    # Feature 3: Cumulative Returns - capture the overall performance over the period
    cumulative_return = (closing_prices[-1] / closing_prices[0]) - 1 if closing_prices[0] != 0 else 0

    # Collect features
    features = [roc, bb_width, cumulative_return]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for relative thresholds
    # Placeholder values for historical stds; these should be derived from actual historical data
    historical_std_risk = 0.2
    historical_std_trend = 0.3
    historical_std_volatility = 0.2

    # Define relative thresholds
    high_risk_threshold = historical_std_risk + 0.5
    low_risk_threshold = historical_std_risk

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
        reward += 20   # Mild positive for SELL
        return np.clip(reward, -100, 100)  # Early exit due to high risk
    elif risk_level > low_risk_threshold:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= low_risk_threshold:
        reward += 30 * np.sign(trend_direction)  # Strongly reward alignment with momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level <= low_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > (historical_std_volatility + 0.5) and risk_level <= low_risk_threshold:
        reward *= 0.5  # Halve the reward magnitude

    return np.clip(reward, -100, 100)