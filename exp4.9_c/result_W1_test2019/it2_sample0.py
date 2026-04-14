import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element)
    days = len(closing_prices)

    # Feature 1: Rate of Change (ROC) - to capture momentum
    roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Bollinger Band Width - to assess volatility
    window = 20
    if days >= window:
        rolling_mean = np.mean(closing_prices[-window:])
        rolling_std = np.std(closing_prices[-window:])
        bb_width = (rolling_std / rolling_mean) if rolling_mean != 0 else 0
    else:
        bb_width = 0

    # Feature 3: Chaikin Money Flow (CMF) - to assess buying and selling pressure
    volumes = s[4:120:6]  # Extract volumes
    money_flow = (closing_prices[1:] - closing_prices[:-1]) * volumes[1:]
    cmf = np.sum(money_flow[-20:]) / np.sum(volumes[-20:]) if np.sum(volumes[-20:]) != 0 else 0

    # Collect features
    features = [roc, bb_width, cmf]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for risk management
    historical_mean_risk = 0.5  # Placeholder: Replace with actual historical mean risk
    historical_std_risk = 0.2    # Placeholder: Replace with actual historical std of risk

    # Define relative thresholds
    high_risk_threshold = historical_mean_risk + 1.5 * historical_std_risk
    low_risk_threshold = historical_mean_risk - 1.5 * historical_std_risk

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
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 25 * np.sign(trend_direction)  # Reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude

    return np.clip(reward, -100, 100)