import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    n = len(closing_prices)
    
    # Feature 1: Price Momentum (last day change)
    if n > 1:
        price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    else:
        price_momentum = 0
    
    # Feature 2: Volatility (standard deviation of closing prices)
    volatility = np.std(closing_prices[-5:])  # Last 5 days
    if n < 5:
        volatility = np.std(closing_prices) if n > 0 else 0
    
    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain) if len(gain) > 0 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss) if len(loss) > 0 else 0
    
    if avg_loss == 0:
        rsi = 100 if avg_gain > 0 else 0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    features = [price_momentum, volatility, rsi]
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
        if enhanced_s[123] > 0:  # BUY-aligned features
            return np.random.uniform(-50, -30)  # Strong negative for buy
        else:  # SELL-aligned features
            return np.random.uniform(5, 10)  # Mild positive for sell

    if risk_level > 0.4:
        if enhanced_s[123] > 0:  # BUY-aligned features
            return np.random.uniform(-20, -10)  # Moderate negative for buy

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] > 0:  # Upward features
            reward += 10  # Positive for upward features
        elif trend_direction < -0.3 and enhanced_s[123] < 0:  # Downward features
            reward += 10  # Positive for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] > 0:  # Oversold (buy)
            reward += 10
        elif enhanced_s[123] < 0:  # Overbought (sell)
            reward += 10

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward