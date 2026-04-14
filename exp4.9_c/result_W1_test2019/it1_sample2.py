import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    days = len(closing_prices)

    # Feature 1: Bollinger Bands
    window = 20
    if days >= window:
        rolling_mean = np.mean(closing_prices[-window:])
        rolling_std = np.std(closing_prices[-window:])
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        price = closing_prices[-1]
        bb_width = (upper_band - lower_band) / rolling_mean  # Width of Bollinger Bands
    else:
        bb_width = 0

    # Feature 2: On-Balance Volume (OBV)
    volumes = s[4:120:6]  # Extract volumes
    obv = np.zeros(days)
    for i in range(1, days):
        if closing_prices[i] > closing_prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closing_prices[i] < closing_prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]

    # Feature 3: Average True Range (ATR)
    high_prices = s[1::6]  # Assuming this is the high prices in the same order
    low_prices = s[2::6]   # Assuming this is the low prices in the same order
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Return the computed features as a numpy array
    features = [bb_width, obv[-1] if days > 0 else 0, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_risk_thresholds = np.array([0.4, 0.7])  # Example thresholds, should be derived from historical data
    risk_threshold_high = historical_risk_thresholds[1]
    risk_threshold_mid = historical_risk_thresholds[0]

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY signals
        reward += 20  # Mild positive for SELL signals
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_mid:
        reward += 30 * np.sign(trend_direction)  # Strongly reward alignment with the trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Halve the reward magnitude

    return np.clip(reward, -100, 100)