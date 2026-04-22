import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window-1] = np.mean(tr[:window-1])
    for i in range(window, len(tr) + 1):
        atr[i - 1] = (atr[i - 2] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:119]
    
    # Calculate additional features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi_14 = np.concatenate((np.full(13, np.nan), calculate_rsi(closing_prices, 14)))  # Pad with NaN
    atr_14 = np.concatenate((np.full(13, np.nan), calculate_atr(high_prices, low_prices, closing_prices, 14)))  # Pad with NaN

    # Create enhanced state
    enhanced_s = np.concatenate([
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        volumes,
        adjusted_closing_prices,
        sma_5[-1:], sma_10[-1:], sma_20[-1:],  # Use the last value of each SMA
        ema_5[-1:], ema_10[-1:],  # Use the last value of each EMA
        rsi_14[-1:],  # Use the last value of RSI
        atr_14[-1:]  # Use the last value of ATR
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    atr = enhanced_s[-1]  # Last value of ATR

    # Calculate historical volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)

    # Define adaptive threshold
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    # Initialize reward
    reward = 0

    # Reward logic based on recent return and thresholds
    if recent_return > threshold:
        reward += 50  # Positive reward for good upward movement
    elif recent_return < -threshold:
        reward -= 50  # Negative reward for bad downward movement
    elif np.abs(recent_return) < threshold:
        reward -= 20  # Penalty for being in a sideways market

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]