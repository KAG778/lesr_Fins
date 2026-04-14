import numpy as np

def revise_state(s):
    # Extract closing prices, high, low, and volumes
    closing_prices = s[0:120:6]  # Closing prices
    high_prices = s[2:120:6]     # High prices
    low_prices = s[3:120:6]      # Low prices
    volumes = s[4:120:6]         # Volumes

    # Feature 1: Average True Range (ATR)
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                         abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR

    # Feature 2: Z-score of the last closing price
    mean_price = np.mean(closing_prices[-20:])  # Mean of last 20 closing prices
    std_price = np.std(closing_prices[-20:])    # Std dev of last 20 closing prices
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0

    # Feature 3: Rate of Change (ROC)
    roc_period = 10
    roc = (closing_prices[-1] - closing_prices[-(roc_period + 1)]) / closing_prices[-(roc_period + 1)] if len(closing_prices) > roc_period else 0

    features = [atr, z_score, roc]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate relative thresholds for risk management
    avg_risk_threshold = 0.5  # This could be determined based on historical data
    if risk_level > avg_risk_threshold:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY-aligned features
        # Mild positive for SELL-aligned features (if applicable)
        reward += np.random.uniform(5, 10)  # MILD POSITIVE for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < avg_risk_threshold:
        if trend_direction > 0 and features[2] > 0:  # Buy signal aligned with upward momentum
            reward += 20
        elif trend_direction < 0 and features[2] < 0:  # Sell signal aligned with downward momentum
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < avg_risk_threshold:
        if features[1] < -1:  # Z-score indicates oversold condition
            reward += 10  # Reward for buying oversold
        elif features[1] > 1:  # Z-score indicates overbought condition
            reward += -10  # Penalty for buying overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < avg_risk_threshold:
        reward *= 0.5  # Reduce magnitude of reward by 50%

    # Ensure reward stays within bounds
    return float(np.clip(reward, -100, 100))