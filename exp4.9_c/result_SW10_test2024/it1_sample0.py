import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Get closing prices (day i at i*6)
    volumes = s[4:120:6]          # Get volumes
    features = []
    
    # 1. Rate of Change (ROC) for price momentum
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(roc)

    # 2. 5-day moving average of close prices
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = closing_prices[-1]
    features.append(moving_average)

    # 3. Volatility based on historical price standard deviation
    historical_volatility = np.std(closing_prices[-20:])  # Standard deviation over the last 20 days
    features.append(historical_volatility)

    # 4. Market breadth indicator (simple example)
    breadth = (np.sum(np.diff(closing_prices) > 0) / len(closing_prices)) if len(closing_prices) > 1 else 0
    features.append(breadth)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk management
    risk_threshold_high = 0.7  # This can also be derived from historical data
    risk_threshold_medium = 0.4

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -40  # Strong negative for BUY-aligned features
        reward += +5   # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < 0:  # Downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features
        reward -= 10  # Penalize breakout-chasing features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward stays within the limits [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward