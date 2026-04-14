import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    
    # Extract closing prices from raw state
    closing_prices = s[0:120:6]  # every 6th element starting from index 0
    
    # Calculate features
    features = []

    # 1. 5-Day Moving Average of Closing Prices
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    features.append(moving_average)

    # 2. Relative Strength Index (RSI) Calculation
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 0  # Not enough data to calculate RSI
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        if loss == 0:
            return 100  # RSI is 100 if there's no loss
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_value = calculate_rsi(closing_prices[-14:])  # Calculate RSI for the last 14 days
    features.append(rsi_value)

    # 3. Volatility Measure (Standard Deviation of Closing Prices)
    volatility = np.std(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    features.append(volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # Positive reward for upward features
        else:
            reward += 10  # Positive reward for downward features (correct bearish bet)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward for mean-reversion features
        reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)