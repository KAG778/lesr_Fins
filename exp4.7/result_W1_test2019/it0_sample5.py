import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12[12:] - ema_26[26:]
    return macd

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.abs(highs[1:] - closes[:-1]), 
                    np.abs(lows[1:] - closes[:-1]))
    atr = np.zeros_like(closes)
    atr[window - 1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # To handle edge cases, pad the results
    sma_5 = np.pad(sma_5, (4, 0), 'constant', constant_values=np.nan)
    sma_10 = np.pad(sma_10, (9, 0), 'constant', constant_values=np.nan)
    sma_20 = np.pad(sma_20, (19, 0), 'constant', constant_values=np.nan)
    ema_5 = np.pad(ema_5, (4, 0), 'constant', constant_values=np.nan)
    rsi = np.pad(np.array([rsi]), (14, 0), 'constant', constant_values=np.nan)
    atr = np.pad(atr, (13, 0), 'constant', constant_values=np.nan)
    
    # Stack all features into the enhanced state (maintaining original order)
    enhanced_s = np.concatenate((closing_prices, opening_prices, high_prices, low_prices, 
                                  volume, closing_prices, sma_5, sma_10, sma_20, ema_5, 
                                  rsi, macd, atr), axis=0)
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    # Calculate historical volatility from closing prices
    historical_vol = np.std(returns)  # Daily volatility in percentage
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    # Determine reward based on recent return against volatility-adaptive threshold
    if recent_return > threshold:
        reward += 50  # Strong positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong negative momentum
    
    # Additional checks based on risk
    if np.abs(recent_return) > threshold:
        reward -= 30  # High risk, penalize
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]