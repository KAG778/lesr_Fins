import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (every 6th element starting from 0)
    
    # Feature 1: Moving Average (MA) over the last 5 days
    ma_period = 5
    ma = np.mean(closing_prices[-ma_period:]) if len(closing_prices) >= ma_period else np.nan
    
    # Feature 2: Relative Strength Index (RSI) - calculating over the last 14 days
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return np.nan
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.abs(np.where(deltas < 0, deltas, 0))
        avg_gain = np.mean(gain[-period:]) if np.sum(gain[-period:]) > 0 else 0
        avg_loss = np.mean(loss[-period:]) if np.sum(loss[-period:]) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices, period=14)
    
    # Feature 3: Price Momentum calculated over the last 5 days
    price_momentum = closing_prices[-1] - closing_prices[-ma_period] if len(closing_prices) >= ma_period else np.nan

    # Combine features into a single array
    features = [ma, rsi, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for buy-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for sell-aligned features
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for buy signals

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds
    return max(-100, min(100, reward))