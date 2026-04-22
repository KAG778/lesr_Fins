import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window - 1] = np.mean(prices[:window])  # Initialize EMA
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

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(abs(highs[1:] - closes[:-1]), 
                               abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window - 1] = np.mean(tr[:window])  # Initialize ATR
    for i in range(window, len(tr)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closes = s[100:119]

    # Calculate features
    sma_5 = np.concatenate((np.full(4, np.nan), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.full(9, np.nan), calculate_sma(closing_prices, 10)))
    rsi = np.concatenate((np.full(13, np.nan), calculate_rsi(closing_prices, 14)))
    atr = np.concatenate((np.full(13, np.nan), calculate_atr(high_prices, low_prices, closing_prices, 14)))

    # Adding calculated features to enhance state
    enhanced_s = np.concatenate((s, sma_5, sma_10, rsi, atr))
    
    # Handle potential NaN values
    enhanced_s = np.nan_to_num(enhanced_s)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return
    
    # Calculate historical volatility
    historical_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(historical_returns)
    
    # Define thresholds based on volatility
    threshold = 2 * historical_vol

    reward = 0
    if recent_return > threshold:
        reward += 50
    elif recent_return < -threshold:
        reward -= 50

    # Incorporate RSI for additional reward logic
    rsi_value = enhanced_s[100]  # Assuming RSI is at index 100 in enhanced_s
    if rsi_value < 30:  # Oversold
        reward += 20
    elif rsi_value > 70:  # Overbought
        reward -= 20

    return np.clip(reward, -100, 100)