import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(prices.shape)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window):
    sma = calculate_sma(prices, window)
    rolling_std = np.std(prices[-window:])  # Calculate std over the last 'window' days
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    return upper_band, lower_band

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-window:])
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closes = s[100:120]

    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Enhanced state
    enhanced_s = np.concatenate((
        s, 
        sma_5[-1:],  # latest SMA 5
        sma_10[-1:],  # latest SMA 10
        ema_5[-1:],  # latest EMA 5
        np.array([rsi_14]),  # latest RSI 14
        np.array([upper_band[-1]]),  # latest upper Bollinger Band
        np.array([lower_band[-1]]),  # latest lower Bollinger Band
        np.array([atr_14])  # latest ATR
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility

    # Calculate the thresholds
    threshold = 2 * historical_vol
    
    reward = 0
    
    # Reward based on recent return and volatility threshold
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Additional conditions can be added based on technical indicators
    rsi = enhanced_s[-5]  # Assuming RSI is the 5th last element
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    return reward