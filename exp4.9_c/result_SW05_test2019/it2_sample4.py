import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    features = []

    # 1. Z-Score of Price Momentum
    recent_momentum = closing_prices[0] - np.mean(closing_prices[1:6]) if len(closing_prices) > 5 else 0
    historical_mean_momentum = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    historical_std_momentum = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1  # Avoid division by zero
    z_momentum = (recent_momentum - historical_mean_momentum) / historical_std_momentum
    features.append(z_momentum)

    # 2. Z-Score of Volatility (Standard Deviation of the last 5 closing prices)
    recent_volatility = np.std(closing_prices[-5:]) if len(closing_prices) > 5 else 0
    historical_mean_volatility = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    historical_std_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1  # Avoid division by zero
    z_volatility = (recent_volatility - historical_mean_volatility) / historical_std_volatility
    features.append(z_volatility)

    # 3. Relative Strength Index (RSI) for trend strength
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0  # Avoid division by zero
        return 100 - (100 / (1 + rs))

    rsi_value = calculate_rsi(closing_prices[-14:])
    features.append(rsi_value)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical standard deviations
    historical_std = np.std(enhanced_s[0:120])  # Using raw state for std calculation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(40, 60)  # Strong negative for BUY
        reward += np.random.uniform(5, 15)    # Mild positive for SELL
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += 15  # Positive reward for upward momentum
        else:  # Downtrend
            reward += 15  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds