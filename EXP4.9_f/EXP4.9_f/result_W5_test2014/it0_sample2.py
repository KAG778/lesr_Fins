import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes
    
    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Average Volume
    average_volume = np.mean(trading_volumes) if len(trading_volumes) > 0 else 0.0
    
    # Feature 3: Price Momentum
    price_momentum = closing_prices[-1] - closing_prices[-2]
    
    features = [price_change_pct, average_volume, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_state[123:]
    reward = 0.0

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming price change percentage > 0 indicates a buy signal
            reward -= np.random.uniform(30, 50)  # Strong negative reward
        else:  # Assuming sell signal or hold
            reward += np.random.uniform(5, 10)  # Mild positive reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= np.random.uniform(10, 20)

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            if features[0] > 0:
                reward += np.random.uniform(10, 20)  # Positive reward for correct buy signal
        elif trend_direction < 0:  # Downtrend
            if features[0] < 0:
                reward += np.random.uniform(10, 20)  # Positive reward for correct sell signal

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Assuming a strong oversold condition
            reward += np.random.uniform(10, 20)  # Reward for buying
        elif features[0] > 0.05:  # Assuming a strong overbought condition
            reward += np.random.uniform(10, 20)  # Reward for selling

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)