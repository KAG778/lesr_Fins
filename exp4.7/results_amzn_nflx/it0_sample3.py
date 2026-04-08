import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window + 1:]:
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

def calculate_macd(prices):
    if len(prices) < 26:
        return np.nan, np.nan
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    signal = calculate_ema(prices, 9)  # Signal line is EMA of MACD
    return macd, signal

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.abs(highs[-window:] - closes[-window:]), 
                    np.abs(lows[-window:] - closes[-window:]))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    enhanced_s = np.copy(s)
    
    # Calculate additional features
    enhanced_s = np.concatenate((enhanced_s, [
        calculate_sma(closing_prices, 5),    # 5-day SMA
        calculate_sma(closing_prices, 10),   # 10-day SMA
        calculate_sma(closing_prices, 20),   # 20-day SMA
        calculate_ema(closing_prices, 5),     # 5-day EMA
        calculate_ema(closing_prices, 10),    # 10-day EMA
        calculate_rsi(closing_prices, 14),    # 14-day RSI
        *calculate_macd(closing_prices),       # MACD and Signal
        calculate_atr(high_prices, low_prices, closing_prices, 14) # 14-day ATR
    ]))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    
    if historical_volatility == 0:
        return 0  # Avoid division by zero

    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    threshold = 2 * historical_volatility  # 2x historical volatility as threshold
    reward = 0

    # Evaluate the reward based on recent returns
    if recent_return > threshold:
        reward += 50  # Good positive return
    elif recent_return < -threshold:
        reward -= 50  # Bad negative return
    
    # Additional conditions based on RSI
    rsi = enhanced_s[100]  # Assuming RSI is the last added feature
    if rsi < 30:
        reward += 20  # Potential buy signal (oversold)
    elif rsi > 70:
        reward -= 20  # Potential sell signal (overbought)

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]