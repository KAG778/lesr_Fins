import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes
    
    # Feature 1: Price Momentum (last closing price - average of last 5 closing prices)
    if len(closing_prices) >= 6:
        momentum = closing_prices[-1] - np.mean(closing_prices[-6:-1])  # Use last 5 days for average
    else:
        momentum = 0.0  # Default to 0 if not enough data

    # Feature 2: Average Daily Volume (last 5 trading days)
    if len(volumes) >= 6:
        avg_volume = np.mean(volumes[-6:-1])  # Average of last 5 days
    else:
        avg_volume = 0.0  # Default to 0 if not enough data

    # Feature 3: Relative Strength Index (RSI) calculation
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 0.0  # Not enough data to calculate RSI
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss > 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices)

    # Return only new features
    return np.array([momentum, avg_volume, rsi])

def intrinsic_reward(enhanced_state):
    # Extract regime information
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= 40  # Example strong penalty
        # MILD POSITIVE reward for SELL-aligned features
        reward += 7   # Example mild reward for selling
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Positive reward for buying in an uptrend
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Positive reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += 10  # Example reward for mean-reversion strategy

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clip reward to range [-100, 100]
    return np.clip(reward, -100, 100)