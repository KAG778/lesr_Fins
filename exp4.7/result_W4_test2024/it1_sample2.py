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
    for price in prices[-window+1:]:
        ema = (price * alpha) + (ema * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window=14):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-1] - lows[-1], 
                    np.maximum(abs(highs[-1] - closes[-2]), abs(lows[-1] - closes[-2])))
    atr = np.mean(tr)
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Enhanced state vector
    enhanced_s = np.copy(s)

    # Calculate indicators
    # Moving Averages
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_10 = calculate_ema(closing_prices, 10)
    
    # RSI
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # Average True Range (ATR)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    volatility = np.std(returns) if len(returns) > 0 else 0
    
    # Append to enhanced state
    enhanced_s = np.concatenate((enhanced_s, [sma_5, sma_10, ema_10, rsi_14, atr, volatility]))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(historical_returns) if len(historical_returns) > 0 else 0

    # Volatility-adaptive threshold
    threshold = 2 * historical_vol
    reward = 0

    # Reward conditions based on recent return and trends
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement
    
    # Trend conditions using SMA
    sma_5 = enhanced_s[-5]  # Last feature added
    sma_10 = enhanced_s[-6]  # Second last feature added
    if sma_5 > sma_10:  # Bullish trend
        reward += 20
    elif sma_5 < sma_10:  # Bearish trend
        reward -= 20

    # Risk penalty based on ATR
    atr = enhanced_s[-4]  # ATR is also one of the last features
    if recent_return < -1.5 * atr:  # Check for significant loss
        reward -= 30  # High risk position
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]