import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    days = len(closing_prices)

    # Feature 1: Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if days > 5 and closing_prices[-5] != 0 else 0

    # Feature 2: Williams %R (14-day)
    highest_high = np.max(closing_prices[-14:]) if days >= 14 else closing_prices[-1]
    lowest_low = np.min(closing_prices[-14:]) if days >= 14 else closing_prices[-1]
    williams_r = ((highest_high - closing_prices[-1]) / (highest_high - lowest_low) * -100) if highest_high != lowest_low else 0

    # Feature 3: Stochastic Oscillator
    if days >= 14:
        k_period = (closing_prices[-1] - lowest_low) / (highest_high - lowest_low) * 100
    else:
        k_period = 0

    # Feature 4: Volume Weighted Average Price (VWAP)
    volumes = s[4:120:6]  # Extract volumes
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0

    # Collect features into a numpy array
    features = [roc, williams_r, k_period, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Historical statistics for relative thresholds
    historical_mean_risk = 0.5  # Replace with actual historical mean risk
    historical_std_risk = 0.2  # Replace with actual historical std of risk
    high_risk_threshold = historical_mean_risk + historical_std_risk
    low_risk_threshold = historical_mean_risk - historical_std_risk

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY
        reward += 20  # Mild positive for SELL
        return np.clip(reward, -100, 100)  # Early exit due to high risk
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative for BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 30 * np.sign(trend_direction)  # Reward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude

    return np.clip(reward, -100, 100)