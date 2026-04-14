import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract volumes

    if len(closing_prices) < 14 or len(volumes) < 14:
        return np.zeros(3)  # Return zeros if there are not enough data points

    # Feature 1: Volatility (Standard Deviation of closing prices over 14 days)
    volatility = np.std(closing_prices[-14:])

    # Feature 2: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices[-14:] * volumes[-14:]) / np.sum(volumes[-14:]) if np.sum(volumes[-14:]) != 0 else 0

    # Feature 3: Average True Range (ATR)
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]  # Extract low prices
    tr = np.maximum(high_prices[-14:] - low_prices[-14:], 
                    np.maximum(np.abs(high_prices[-14:] - closing_prices[-14:]), 
                               np.abs(low_prices[-14:] - closing_prices[-14:])))
    atr = np.mean(tr)  # Average True Range

    features = [volatility, vwap, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds for risk level
    historical_risk = np.std(enhanced_s[123:126])  # Assuming features are in the context of risk
    risk_threshold = historical_risk * 0.7  # Example relative threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > (risk_threshold / 2):
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < (risk_threshold / 2):
        if trend_direction > 0:
            reward += 30  # Positive reward for correct trend following in uptrend
        else:
            reward += 30  # Positive reward for correct trend following in downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < (risk_threshold / 2):
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < (risk_threshold / 2):
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward