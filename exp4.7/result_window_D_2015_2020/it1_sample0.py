import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))
    return np.mean(true_ranges[-window:])  # Return average true range

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Calculate features
    sma_5 = np.concatenate((np.array([np.nan]*4), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.array([np.nan]*9), calculate_sma(closing_prices, 10)))
    sma_20 = np.concatenate((np.array([np.nan]*19), calculate_sma(closing_prices, 20)))
    
    ema_5 = np.concatenate((np.array([np.nan]*4), calculate_ema(closing_prices, 5)))
    ema_10 = np.concatenate((np.array([np.nan]*9), calculate_ema(closing_prices, 10)))
    
    rsi = np.concatenate((np.array([np.nan]*13), np.array([calculate_rsi(closing_prices, 14)])))
    atr = np.concatenate((np.array([np.nan]*13), np.array([calculate_atr(high_prices, low_prices, closing_prices, 14)])))
    
    # Combine features into enhanced state
    enhanced_s = np.concatenate([s, sma_5, sma_10, sma_20, ema_5, ema_10, rsi, atr])
    
    return enhanced_s