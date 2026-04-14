import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0:120:6]  # Every 6th element starting from index 0 is the closing price
    
    # Feature 1: Relative Strength Index (RSI)
    def rsi(prices, period=14):
        if len(prices) < period:
            return 0
        delta = np.diff(prices[-period:])
        gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
        loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
        rs = gain / loss if loss else 0
        return 100 - (100 / (1 + rs))
    
    features.append(rsi(closing_prices))
    
    # Feature 2: Average True Range (ATR)
    def atr(prices, period=14):
        if len(prices) < period:
            return 0
        high = s[2:120:6]
        low = s[3:120:6]
        close = s[0:120:6]
        tr = np.maximum(high[-period:] - low[-period:], np.maximum(np.abs(high[-period:] - close[-period-1:-1]), np.abs(low[-period:] - close[-period-1:-1])))
        return np.mean(tr)
    
    features.append(atr(closing_prices))
    
    # Feature 3: Z-score of current price relative to historical mean
    mean_price = np.mean(closing_prices[-20:])  # Last 20 days
    std_dev = np.std(closing_prices[-20:]) if np.std(closing_prices[-20:]) > 0 else 1  # Avoid division by zero
    z_score = (closing_prices[-1] - mean_price) / std_dev
    features.append(z_score)
    
    # Feature 4: Exponential Moving Average (EMA)
    def ema(prices, span=20):
        if len(prices) < span:
            return 0
        return np.mean(prices[-span:])  # Simple approach, can also use exponential weighting for better results
    
    features.append(ema(closing_prices))
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50 * (risk_level - 0.7)  # Strong negative for BUY-aligned features
        if trend_direction < 0:
            reward += 10 * (1 - risk_level)  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20 * (risk_level - 0.4)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 25 * abs(trend_direction)  # Reward momentum alignment based on strength of trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]