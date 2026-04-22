import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    ema = np.zeros_like(prices)
    ema[:window] = np.nan  # First `window` values are NaN
    ema[window-1] = np.mean(prices[:window])  # First EMA is the SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * (2/(window + 1)) + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.abs(np.where(deltas < 0, deltas, 0))
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12[25:] - ema_26[25:]  # Align the lengths
    return macd

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window-1] = np.mean(tr[:window])  # First ATR is the SMA of TR
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i-1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volumes = s[80:99]
    adjusted_closing_prices = s[100:119]

    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)

    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Handle edge cases for arrays of different lengths
    features = [
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        trading_volumes,
        adjusted_closing_prices,
        sma_5, sma_10, sma_20,
        ema_5, ema_10,
        rsi,
        macd,
        atr
    ]

    # Flatten the features into a single array and handle NaNs
    enhanced_s = np.concatenate([f[np.newaxis, :] if f.ndim == 1 else f for f in features], axis=0)
    enhanced_s = np.nan_to_num(enhanced_s)  # Replace NaNs with zeros

    return enhanced_s.flatten()

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Percentage return

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    # Reward based on recent return relative to historical volatility
    if recent_return < -threshold:
        reward -= 50  # Negative reward for large negative return
    elif recent_return > threshold:
        reward += 50  # Positive reward for large positive return

    # Additional reward based on RSI for oversold/overbought conditions
    rsi = enhanced_s[112]  # Assuming RSI is at index 112
    if rsi < 30:  # Oversold condition
        reward += 20
    elif rsi > 70:  # Overbought condition
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]