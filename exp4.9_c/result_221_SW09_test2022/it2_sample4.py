import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Exponential Moving Average (EMA) of closing prices
    ema_period = 14
    ema = np.mean(closing_prices[-ema_period:]) if len(closing_prices) >= ema_period else 0

    # Feature 2: On-Balance Volume (OBV)
    obv = np.zeros_like(closing_prices)
    for i in range(1, len(closing_prices)):
        if closing_prices[i] > closing_prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif closing_prices[i] < closing_prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    obv_value = obv[-1] if len(obv) > 0 else 0

    # Feature 3: Rate of Change (ROC)
    roc_period = 14  # 14-day ROC
    roc = ((closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period]) * 100 if len(closing_prices) >= roc_period else 0

    # Feature 4: Williams %R (a momentum indicator)
    highest_high = np.max(closing_prices[-14:]) if len(closing_prices) >= 14 else 0
    lowest_low = np.min(closing_prices[-14:]) if len(closing_prices) >= 14 else 0
    williams_r = ((highest_high - closing_prices[-1]) / (highest_high - lowest_low) * -100) if (highest_high - lowest_low) != 0 else 0

    features = [ema, obv_value, roc, williams_r]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds for rewards based on historical data
    historical_volatility = np.std(enhanced_s[0:120])  # Assuming we can derive the historical volatility from the raw state
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_moderate = 0.4 * historical_volatility
    trend_threshold = 0.3 * historical_volatility
    volatility_threshold = 0.6 * historical_volatility

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY
        reward += np.random.uniform(10, 20)   # Mild positive for SELL
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(5, 15)  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        reward += np.random.uniform(10, 20)  # Positive reward for momentum alignment

    # Priority 3: Sideways Market / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion strategies

    # Priority 4: High Volatility
    if volatility_level > volatility_threshold and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds of [-100, 100]
    reward = max(-100, min(100, reward))

    return reward