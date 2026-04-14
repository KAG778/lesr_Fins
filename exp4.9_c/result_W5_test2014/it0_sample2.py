import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    # Feature 1: Price Momentum (current close - previous close)
    price_momentum = s[0] - s[6]  # Current close (day 19) - Close 20 days ago (day 0)
    
    # Feature 2: Price Relative Strength Index (RSI) over the last 14 days
    # Calculate RSI using a custom function
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    prices = s[::6]  # Extracting only the closing prices
    rsi = calculate_rsi(prices)

    # Feature 3: Moving Average Convergence Divergence (MACD)
    # Short-term EMA (12 days) and Long-term EMA (26 days)
    def calculate_macd(prices):
        short_ema = np.mean(prices[-12:]) if len(prices) >= 12 else 0
        long_ema = np.mean(prices[-26:]) if len(prices) >= 26 else 0
        return short_ema - long_ema

    macd = calculate_macd(prices)

    features = [price_momentum, rsi, macd]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    
    # Setting initial reward
    reward = 0.0

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40 if features[0] > 0 else 0  # Assuming feature[0] relates to BUY signal
        # Mild positive reward for SELL-aligned features
        reward += 5 if features[0] < 0 else 0  # Assuming feature[0] relates to SELL signal

    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20 if features[0] > 0 else 0  # Assuming feature[0] relates to BUY signal

    # **Priority 2 — TREND FOLLOWING**
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Assuming feature[0] relates to upward signal
            reward += 10
        elif trend_direction < -0.3 and features[0] < 0:  # Assuming feature[0] relates to downward signal
            reward += 10

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Assuming feature[1] relates to RSI (oversold)
            reward += 10  # Reward buying in oversold conditions
        elif features[1] > 70:  # Assuming feature[1] relates to RSI (overbought)
            reward += 10  # Reward selling in overbought conditions

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds