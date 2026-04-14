import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract the closing prices to compute features
    closing_prices = s[0:120:6]  # Closing prices are at indices 0, 6, 12, ..., 114

    # Feature 1: Price momentum (current close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0.0
    
    # Feature 2: Moving Average (Simple Moving Average over the last 5 days)
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    
    # Feature 3: Relative Strength Index (RSI) calculation
    gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
    losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
    
    average_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.0
    average_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.0
    
    rs = average_gain / average_loss if average_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs)) if average_gain + average_loss > 0 else 0.0
    
    # Handle edge cases (for example, when not enough data is available)
    features = [price_momentum, moving_average, rsi]
    
    return np.array(features)

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
        reward -= 40.0  # Strong negative for buying in dangerous conditions
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for buying in elevated risk conditions

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if len(features) > 0:
            if trend_direction > 0:  # Uptrend
                reward += features[0] * 5.0  # Price momentum
            else:  # Downtrend
                reward += -features[0] * 5.0  # Price momentum (inverted for bearish)

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # RSI < 30 indicates oversold condition
            reward += 10.0  # Positive for buying in oversold condition
        elif features[2] > 70:  # RSI > 70 indicates overbought condition
            reward += 10.0  # Positive for selling in overbought condition

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))