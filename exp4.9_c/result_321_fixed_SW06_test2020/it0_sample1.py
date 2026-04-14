import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes
    
    # Feature 1: Price Change from two days ago
    price_change = closing_prices[-1] - closing_prices[-3] if len(closing_prices) > 2 else 0.0
    
    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    
    # Feature 3: Relative Strength Index (RSI) - using a typical 14-day period for RSI calculation
    rsi_period = 14
    if len(closing_prices) >= rsi_period:
        delta = closing_prices[-rsi_period:] - closing_prices[-rsi_period-1:-1]
        gain = np.where(delta > 0, delta, 0).mean()
        loss = -np.where(delta < 0, delta, 0).mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if not enough data

    return np.array([price_change, avg_volume, rsi])

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive price change
            reward += trend_direction * 10.0  # Reward for correct trend
        else:  # Negative price change
            reward += -trend_direction * 10.0  # Reward for contrarian bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Oversold condition
            reward += 5.0  # Buy signal for mean reversion
        elif features[2] > 70:  # Overbought condition
            reward += 5.0  # Sell signal for mean reversion

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))