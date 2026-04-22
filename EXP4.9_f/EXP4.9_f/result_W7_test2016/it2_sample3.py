import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Average True Range (ATR) for volatility measurement
    def calculate_atr(prices, volumes, period=14):
        tr = np.zeros(len(prices) - 1)
        for i in range(1, len(prices)):
            tr[i - 1] = max(prices[i] - prices[i - 1], 
                            abs(prices[i] - prices[i - 1]), 
                            abs(prices[i - 1] - prices[i - 1]))
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr) if len(tr) > 0 else 0
        return atr

    atr = calculate_atr(closing_prices, volumes)
    features.append(atr)

    # Feature 2: On-Balance Volume (OBV)
    def calculate_obv(closing_prices, volumes):
        obv = np.zeros(len(closing_prices))
        for i in range(1, len(closing_prices)):
            if closing_prices[i] > closing_prices[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif closing_prices[i] < closing_prices[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]
        return obv[-1] if len(obv) > 0 else 0

    obv = calculate_obv(closing_prices, volumes)
    features.append(obv)

    # Feature 3: Stochastic Oscillator
    def calculate_stochastic(prices, period=14):
        if len(prices) < period:
            return 0, 0  # Not enough data
        lowest_low = np.min(prices[-period:])
        highest_high = np.max(prices[-period:])
        stochastic = (prices[-1] - lowest_low) / (highest_high - lowest_low + 1e-10) * 100  # %K
        return stochastic
    
    stochastic = calculate_stochastic(closing_prices)
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
            reward += 30  # Strong positive for upward features
        elif trend_direction < -trend_threshold:
            reward += 30  # Strong positive for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward