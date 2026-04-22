import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    trading_volumes = s[4::6]  # Trading volumes
    
    features = []
    
    # Feature 1: 14-day Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 2: 14-day Average True Range (ATR) - Measures market volatility
    high_low = closing_prices[1:] - closing_prices[:-1]
    high_close = np.abs(closing_prices[1:] - closing_prices[:-1])
    low_close = np.abs(closing_prices[1:] - closing_prices[:-1])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    features.append(atr)

    # Feature 3: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * trading_volumes) / np.sum(trading_volumes)
    features.append(vwap)

    # Feature 4: Price Momentum (current close vs. close 5 days ago)
    momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if len(closing_prices) > 5 else 0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0

    # Calculate dynamic thresholds based on historical standard deviation
    avg_risk = np.mean(features[1])  # Assuming feature[1] relates to volatility (e.g., ATR)
    std_risk = np.std(features[1])
    
    risk_threshold_high = avg_risk + 1.5 * std_risk  # High-risk threshold
    risk_threshold_medium = avg_risk + 0.5 * std_risk  # Medium-risk threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong penalty for BUY-aligned features
        reward -= 40
        # Mild positive reward for SELL-aligned features
        if features[3] < 0:  # Assuming feature[3] indicates a bearish signal
            reward += 10
    elif risk_level > risk_threshold_medium:
        # Moderate penalty for BUY signals
        reward -= 20

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:
            reward += 10 * features[3]  # Positive reward for upward momentum
        else:
            reward += 10 * -features[3]  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < -0.1:  # Assuming negative values indicate oversold condition
            reward += 15  # Reward for buying in oversold condition
        elif features[3] > 0.1:  # Assuming positive values indicate overbought condition
            reward -= 15  # Penalty for buying in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range