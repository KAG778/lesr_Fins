import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    num_days = len(closing_prices)
    
    # Feature 1: 5-Day Moving Average
    moving_average = np.mean(closing_prices[-5:]) if num_days >= 5 else np.nan
    
    # Feature 2: Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0]) if len(deltas[deltas > 0]) > 0 else 0
        loss = -np.mean(deltas[deltas < 0]) if len(deltas[deltas < 0]) > 0 else 0
        rs = gain / loss if loss > 0 else 0
        return 100 - (100 / (1 + rs))
    
    rsi = calculate_rsi(closing_prices) if num_days > 14 else np.nan
    
    # Feature 3: Price Change - Percentage change from the previous day
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if num_days >= 2 else np.nan

    features = []
    
    # Append features, handling edge cases (NaN)
    features.append(moving_average if not np.isnan(moving_average) else 0)
    features.append(rsi if not np.isnan(rsi) else 0)
    features.append(price_change if not np.isnan(price_change) else 0)

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
        reward -= 40.0  # Strong negative for BUY
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if len(features) > 0:
            reward += trend_direction * features[0] * 5.0  # Moving average affecting reward

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # RSI < 30: Oversold, consider BUY
            reward += 10.0  # Positive for oversold
        elif features[1] > 70:  # RSI > 70: Overbought, consider SELL
            reward += 10.0  # Positive for overbought
    
    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))