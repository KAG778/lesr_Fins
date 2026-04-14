import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Rate of Change (ROC) for the last 14 days
    roc = (s[0] - s[84]) / s[84] if s[84] != 0 else 0  # (Close day 19 - Close day 5) / Close day 5
    features.append(roc)

    # Feature 2: Average True Range (ATR) over the last 14 days
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    closes = s[0:120:6]       # Extract closing prices
    
    def calculate_atr(highs, lows, closes, period=14):
        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                   np.abs(lows[1:] - closes[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    atr = calculate_atr(high_prices, low_prices, closes)
    features.append(atr)

    # Feature 3: Z-score of the last 14 closing prices for mean reversion
    mean_price = np.mean(closes[-14:]) if len(closes) >= 14 else 0
    std_price = np.std(closes[-14:]) if len(closes) >= 14 else 0
    z_score = (closes[-1] - mean_price) / std_price if std_price != 0 else 0
    features.append(z_score)

    # Feature 4: Correlation between price changes and volume changes
    price_changes = np.diff(closes)
    volume_changes = np.diff(s[4:120:6])  # Volumes are at every 6th element, starting from index 4
    if len(price_changes) > 0 and len(volume_changes) > 0:
        correlation = np.corrcoef(price_changes[-10:], volume_changes[-10:])[0, 1]
    else:
        correlation = 0
    features.append(correlation)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate historical thresholds based on past risk levels
    historical_risk_levels = np.array([enhanced_s[i] for i in range(123, len(enhanced_s)) if i % 6 == 4])
    mean_risk_level = np.mean(historical_risk_levels) if len(historical_risk_levels) > 0 else 0
    std_risk_level = np.std(historical_risk_levels) if len(historical_risk_levels) > 0 else 1  # Avoid division by zero

    high_risk_threshold = mean_risk_level + 1.5 * std_risk_level
    mid_risk_threshold = mean_risk_level + 0.5 * std_risk_level

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > mid_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level <= mid_risk_threshold:
        if trend_direction > 0:
            reward += 20  # Reward for upward trend
        else:
            reward += 20  # Reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < mid_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:  # Using relative volatility threshold
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)