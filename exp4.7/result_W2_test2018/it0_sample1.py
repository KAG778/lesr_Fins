import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    return ema_12[-len(ema_26):] - ema_26

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i-1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Enhanced state array
    enhanced_s = np.zeros(120 + 8)  # Original 120 dimensions + new features
    
    # Add original state
    enhanced_s[0:120] = s
    
    # Calculate new features
    enhanced_s[120] = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan  # 5-day SMA
    enhanced_s[121] = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan  # 10-day SMA
    enhanced_s[122] = calculate_ema(closing_prices, 12)[-1]  # 12-day EMA
    enhanced_s[123] = calculate_rsi(closing_prices, 14)  # 14-day RSI
    enhanced_s[124] = calculate_macd(closing_prices)[-1]  # MACD
    enhanced_s[125] = calculate_atr(high_prices, low_prices, closing_prices, 14)[-1]  # 14-day ATR
    enhanced_s[126] = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    enhanced_s[127] = np.mean(volumes[-5:])  # 5-day average volume

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return
    volatility = enhanced_s[126]  # Historical volatility

    threshold = 2 * volatility  # Adapt threshold based on historical volatility
    reward = 0

    # Reward based on recent return and volatility
    if recent_return > threshold:  # Strong positive return
        reward += 50
    elif recent_return < -threshold:  # Strong negative return
        reward -= 50

    # Add other conditions based on RSI and trend indicators
    rsi = enhanced_s[123]
    if rsi < 30:  # Oversold condition
        reward += 10
    elif rsi > 70:  # Overbought condition
        reward -= 10

    # Check if the price is above the 5-day SMA (indicating an uptrend)
    if closing_prices[-1] > enhanced_s[120]:  
        reward += 10
    else:  # Otherwise, consider it less favorable
        reward -= 10

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]