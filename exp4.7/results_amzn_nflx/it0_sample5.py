import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    return ema12[-len(ema26):] - ema26

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.convolve(tr, np.ones(window)/window, mode='valid')
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    
    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # Handle edge cases
    sma_5 = np.pad(sma_5, (len(closing_prices) - len(sma_5), 0), 'edge')
    sma_10 = np.pad(sma_10, (len(closing_prices) - len(sma_10), 0), 'edge')
    rsi = np.pad(np.array([rsi]), (len(closing_prices) - 1, 0), 'edge')[0]
    macd = np.pad(macd, (len(closing_prices) - len(macd), 0), 'edge')
    atr = np.pad(atr, (len(closing_prices) - len(atr), 0), 'edge')

    # Construct enhanced state
    enhanced_s = np.concatenate((
        s, 
        sma_5, 
        sma_10, 
        np.array([rsi]), 
        macd, 
        atr
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    recent_return = returns[-1] if len(returns) > 0 else 0

    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 0 else 1
    threshold = 2 * historical_vol  # Relative threshold
    
    reward = 0
    
    # Reward logic based on recent return and volatility
    if recent_return > threshold:
        reward += 50  # Strong upward move
    elif recent_return < -threshold:
        reward -= 50  # Strong downward move
    
    # Consider trend based on moving averages
    sma_5 = enhanced_s[120:139]
    sma_10 = enhanced_s[139:159]
    
    if sma_5[-1] > sma_10[-1]:
        reward += 20  # Uptrend
    elif sma_5[-1] < sma_10[-1]:
        reward -= 20  # Downtrend
    
    return np.clip(reward, -100, 100)