import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Average True Range (ATR) over the last 14 days
    def calculate_atr(prices, volumes, period=14):
        if len(prices) < period:
            return 0
        tr = np.maximum(prices[-1] - prices[-2], 
                       np.maximum(prices[-1] - prices[-period:-1],
                                  prices[-period:-1] - prices[-1]))
        atr = np.mean(tr)
        return atr
    
    atr = calculate_atr(closing_prices, volumes)
    features.append(atr)

    # Feature 2: Rate of Change (ROC) over the last 14 days
    if len(closing_prices) >= 14:
        roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15]  # Percentage change
    else:
        roc = 0
    features.append(roc)

    # Feature 3: Stochastic Oscillator
    if len(closing_prices) >= 14:
        lowest_low = np.min(closing_prices[-14:])
        highest_high = np.max(closing_prices[-14:])
        stochastic = (closing_prices[-1] - lowest_low) / (highest_high - lowest_low + 1e-10)  # Avoid division by zero
    else:
        stochastic = 0
    features.append(stochastic)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[0:120])  # Use the raw state for variability
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10 if enhanced_s[123] < 0 else 0  # Mild positive for SELL-aligned features
    elif risk_level > 0.4 * historical_std:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold:
            reward += 30  # Strong positive reward for upward features
        elif trend_direction < -trend_threshold:
            reward += 30  # Strong positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward