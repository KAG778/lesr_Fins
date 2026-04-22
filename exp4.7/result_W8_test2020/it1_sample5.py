import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else np.mean(gain)
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else np.mean(loss)
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12[11:] - ema_26[25:]  # Align lengths
    return np.concatenate((np.nan * np.ones(25), macd))

def calculate_bollinger_bands(prices, window):
    sma = calculate_sma(prices, window)
    rolling_std = np.std(prices[-window:])  # Calculate std over the last 'window' days
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    return upper_band, lower_band

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-window:])  # Average True Range
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    boll_upper, boll_lower = calculate_bollinger_bands(closing_prices, 20)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)
    macd = calculate_macd(closing_prices)

    # Create enhanced state
    enhanced_s = np.concatenate((
        s, 
        sma_5[-1:],  # latest SMA 5
        sma_10[-1:],  # latest SMA 10
        sma_20[-1:],  # latest SMA 20
        ema_5[-1:],  # latest EMA 5
        np.array([rsi_14]),  # latest RSI 14
        np.array([boll_upper[-1]]),  # latest upper Bollinger Band
        np.array([boll_lower[-1]]),  # latest lower Bollinger Band
        np.array([atr_14]),  # latest ATR
        macd[-1:]  # latest MACD
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_volatility = np.std(returns)  # Daily volatility in percentage
    
    # Set thresholds based on historical volatility
    threshold = 2 * historical_volatility  # Volatility-adaptive threshold

    reward = 0

    # Determine reward based on recent return and volatility thresholds
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Adjust reward based on RSI
    rsi = enhanced_s[-5]  # Assuming RSI is the 5th last element
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]