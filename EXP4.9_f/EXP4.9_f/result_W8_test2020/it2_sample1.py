import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]      # Extract high prices
    low_prices = s[3::6]       # Extract low prices
    volume = s[4::6]           # Extract volume

    # Feature 1: Rate of Change (ROC) - measures the percentage change in price
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Feature 2: Exponential Moving Average (EMA) - gives more weight to recent prices
    ema = np.mean(closing_prices[-14:]) if len(closing_prices) >= 14 else 0  # Average over the last 14 days

    # Feature 3: Chaikin Money Flow (CMF) - measures the buying and selling pressure for a specified period
    money_flow_multiplier = (closing_prices[-1] - low_prices[-1]) - (high_prices[-1] - closing_prices[-1])
    money_flow_volume = money_flow_multiplier * volume[-1]
    cmf = np.sum(money_flow_volume) / np.sum(volume[-14:]) if np.sum(volume[-14:]) != 0 else 0  # Average over the last 14 days

    features = [roc, ema, cmf]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Determine thresholds based on historical standard deviation of risk_level
    risk_std = np.std(enhanced_s[120:123])  # Assuming the regime vector is updated over time
    high_risk_threshold = 0.7 * risk_std
    medium_risk_threshold = 0.4 * risk_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY signals in high risk
    elif risk_level > medium_risk_threshold:
        reward += 20  # Mildly positive reward for SELL signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        momentum_reward = 20 * np.clip(trend_direction, 0, 1)  # Reward momentum alignment
        reward += momentum_reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < medium_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within specified bounds