import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate daily returns
    daily_returns = np.zeros(20)
    for i in range(20):
        closing_price = s[i * 6 + 0]
        previous_closing_price = s[(i - 1) * 6 + 0] if i > 0 else closing_price
        daily_returns[i] = (closing_price - previous_closing_price) / previous_closing_price if previous_closing_price != 0 else 0
    
    # Feature 1: Average daily return over the last 20 days
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)
    
    # Feature 2: Standard deviation of daily returns (volatility)
    volatility = np.std(daily_returns)
    features.append(volatility)
    
    # Feature 3: Price momentum (current closing price vs. closing price 5 days ago)
    price_momentum = (s[0] - s[5 * 6 + 0]) / s[5 * 6 + 0] if s[5 * 6 + 0] != 0 else 0
    features.append(price_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        if features[0] > 0:  # Assume feature[0] indicates a bullish signal
            return -40.0  # Strong penalty for buying in dangerous conditions
        else:  # Selling aligned features
            return 8.0  # Mild positive reward for selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            return -15.0  # Penalty for buying under elevated risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            return max(10 * features[0], 10)  # Positive reward for long positions
        else:
            return max(10 * -features[0], 10)  # Positive reward for short positions

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        if features[0] < -0.1:  # Assuming negative values indicate oversold condition
            return 15.0  # Positive reward for buying in oversold condition
        elif features[0] > 0.1:  # Assuming positive values indicate overbought condition
            return -15.0  # Negative reward for buying in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        # Reduce reward magnitude by 50%
        return reward * 0.5  # Reduce the overall reward by half in high volatility conditions

    return reward  # Default return if no conditions are met