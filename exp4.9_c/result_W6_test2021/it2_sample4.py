import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Rate of Change (ROC) for the last 14 days
    roc = (s[0:120:6][-1] - s[0:120:6][-14]) / s[0:120:6][-14] if s[0:120:6][-14] != 0 else 0
    features.append(roc)

    # Feature 2: Average True Range (ATR) over the last 14 days
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - s[0:120:6][:-1]), 
                                        np.abs(low_prices[1:] - s[0:120:6][:-1]))))
    features.append(atr)

    # Feature 3: Z-score of the last 14 closing prices for mean reversion
    closing_prices = s[0:120:6][-14:]  # Extract last 14 closing prices
    mean_price = np.mean(closing_prices)
    std_price = np.std(closing_prices)
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    features.append(z_score)

    # Feature 4: Correlation between price change and volume change over the last 14 days
    price_changes = np.diff(s[0:120:6])[-14:]  # Last 14 price changes
    volume_changes = np.diff(s[4:120:6])[-14:]  # Last 14 volume changes
    if len(price_changes) > 0 and len(volume_changes) > 0:
        correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
    else:
        correlation = 0
    features.append(correlation)

    # Feature 5: Momentum (5-day Price Momentum)
    try:
        price_momentum = (s[0:120:6][-1] - s[0:120:6][-6]) / s[0:120:6][-6]  # Last 5-day momentum
    except ZeroDivisionError:
        price_momentum = 0.0
    features.append(price_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate relative thresholds based on historical data
    feature_std = np.std(enhanced_s[123:])  # Use standard deviation of features for dynamic thresholds
    high_risk_threshold = 0.7 * feature_std
    medium_risk_threshold = 0.4 * feature_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY signals
        reward += 10   # Mild positive reward for SELL signals
    elif risk_level > medium_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= medium_risk_threshold:
        reward += 20 if trend_direction > 0 else 20  # Reward for trend alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < medium_risk_threshold:
        reward += 15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * feature_std:  # Relative threshold based on volatility
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds