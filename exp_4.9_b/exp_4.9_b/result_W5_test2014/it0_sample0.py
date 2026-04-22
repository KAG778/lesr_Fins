import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices[-window:])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                   np.abs(highs[-window:] - closes[-window:-1]), 
                   np.abs(lows[-window:] - closes[-window:-1]))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closes = s[100:120]
    
    # Feature extraction
    features = []
    
    # Multi-timeframe Trend Indicators
    features.append(calculate_sma(closing_prices, 5))  # 5-day SMA
    features.append(calculate_sma(closing_prices, 10)) # 10-day SMA
    features.append(calculate_sma(closing_prices, 20)) # 20-day SMA
    features.append(calculate_ema(closing_prices, 5))  # 5-day EMA
    features.append(calculate_ema(closing_prices, 10)) # 10-day EMA
    features.append(closing_prices[-1] / calculate_sma(closing_prices, 10)) # Price relative to 10-day SMA
    
    # Momentum Indicators
    features.append(calculate_rsi(closing_prices, 5))   # 5-day RSI
    features.append(calculate_rsi(closing_prices, 10))  # 10-day RSI
    macd_line = calculate_ema(closing_prices, 12) - calculate_ema(closing_prices, 26)  # MACD line
    signal_line = calculate_ema(np.concatenate([closing_prices, np.array([macd_line])]), 9)  # Signal line
    features.append(macd_line)  # MACD line
    features.append(signal_line) # Signal line
    features.append(macd_line - signal_line)  # MACD histogram
    
    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices))  # 5-day historical volatility
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]))  # 20-day historical volatility
    features.append(historical_volatility_5) 
    features.append(historical_volatility_20)
    features.append(historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else np.nan)  # Volatility ratio
    
    # Volume-Price Relationship
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], 
                          np.where(np.diff(closing_prices) < 0, -volumes[1:], 0)))  # On-Balance Volume
    features.append(obv)
    features.append(volumes[-1] / np.mean(volumes[-20:]))  # Volume ratio
    features.append(np.corrcoef(volumes[-20:], closing_prices[-20:])[0, 1])  # Volume-price correlation

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else np.nan
    trend_strength = np.nan  # Placeholder for trend strength calculation
    if len(closing_prices) > 1:
        x = np.arange(len(closing_prices))
        coeffs = np.polyfit(x, closing_prices, 1)
        trend_strength = np.corrcoef(closing_prices, x)[0, 1] ** 2  # R² for trend strength
    features.append(volatility_ratio)
    features.append(trend_strength)
    features.append(closing_prices[-1] / np.max(closing_prices[-20:]))  # Price position in range
    features.append(np.mean(volumes[-5:]) / np.mean(volumes[-20:]))  # Volume ratio regime

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(features)))
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    historical_vol = np.std(np.diff(closing_prices))  # Daily returns
    
    reward = 0
    
    # Determine reward based on position
    if position == 0:  # Not holding
        if recent_return > 2 * historical_vol:  # Strong buy signal threshold
            reward += 50
        elif recent_return < -2 * historical_vol:  # Strong sell signal threshold
            reward -= 50
    else:  # Holding
        if recent_return > 0:  # Positive return while holding
            reward += 10
        elif recent_return < -2 * historical_vol:  # Significant drop while holding
            reward -= 50
            
    return np.clip(reward, -100, 100)  # Clip reward to range [-100, 100]