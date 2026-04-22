import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    return np.convolve(prices, weights, mode='valid')

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
    macd = ema_12[-len(ema_26):] - ema_26
    signal = calculate_ema(macd, 9)  # Signal line is EMA of MACD
    return macd, signal

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(tr)
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]

    # Calculate indicators
    sma_5 = np.pad(calculate_sma(closing_prices, 5), (4, 0), 'constant', constant_values=np.nan)
    sma_10 = np.pad(calculate_sma(closing_prices, 10), (9, 0), 'constant', constant_values=np.nan)
    ema_5 = np.pad(calculate_ema(closing_prices, 5), (4, 0), 'constant', constant_values=np.nan)
    rsi_14 = np.pad(np.array([calculate_rsi(closing_prices[i:i+14], 14) for i in range(len(closing_prices)-14)]), (13, 0), 'constant', constant_values=np.nan)[-20:]
    macd, signal = calculate_macd(closing_prices)
    atr_14 = np.pad(calculate_atr(high_prices, low_prices, closing_prices, 14), (13, 0), 'constant', constant_values=np.nan)

    # Combine features into new state
    enhanced_s = np.concatenate((s, sma_5, sma_10, ema_5, rsi_14, macd, signal, atr_14))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    rsi = enhanced_s[-7]  # Last RSI value (assuming it's the 7th last feature)
    
    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 1e-10  # Avoid division by zero

    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol

    reward = 0

    # Reward based on recent return
    if recent_return > threshold:
        reward += 50  # High return
    elif recent_return < -threshold:
        reward -= 50  # Significant loss

    # Penalize if RSI indicates overbought or oversold
    if rsi > 70:
        reward -= 30  # Overbought
    elif rsi < 30:
        reward += 30  # Oversold (potential buy signal)

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)