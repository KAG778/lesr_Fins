import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Price Momentum
    # Calculate momentum as the change in closing price from day -1 to day -2
    price_momentum = closing_prices[-1] - closing_prices[-2]
    if closing_prices[-2] != 0:
        price_momentum_normalized = price_momentum / closing_prices[-2]
    else:
        price_momentum_normalized = 0  # Handle division by zero
    features.append(price_momentum_normalized)

    # Feature 2: Relative Strength Index (RSI)
    # Calculate RSI over the last 14 days (or fewer if not enough data)
    delta = closing_prices[1:] - closing_prices[:-1]
    gain = np.where(delta > 0, delta, 0).mean()
    loss = np.abs(np.where(delta < 0, delta, 0)).mean()
    
    if (gain + loss) != 0:
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if there's no gain or loss
    features.append(rsi)

    # Feature 3: Volume Change
    # Calculate percentage change in volume from day -1 to day -2
    volume_change = volumes[-1] - volumes[-2]
    if volumes[-2] != 0:
        volume_change_normalized = volume_change / volumes[-2]
    else:
        volume_change_normalized = 0  # Handle division by zero
    features.append(volume_change_normalized)

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
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        # Here you might consider further logic based on your features
        return reward
    
    if risk_level > 0.4:
        reward += -10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming we have some logic to determine mean-reversion signals
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward