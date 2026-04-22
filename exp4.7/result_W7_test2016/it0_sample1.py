import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    alpha = 2 / (window + 1)
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    return ema_12[-len(ema_26):] - ema_26

def calculate_bollinger_bands(prices, window):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, window)
    rolling_std = np.std(prices[-window:])
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    return upper_band, lower_band

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:119]

    # Create enhanced state
    enhanced_s = np.concatenate([s])

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    if len(sma_5) > 0:
        enhanced_s = np.append(enhanced_s, sma_5[-1])
    if len(sma_10) > 0:
        enhanced_s = np.append(enhanced_s, sma_10[-1])
    if len(sma_20) > 0:
        enhanced_s = np.append(enhanced_s, sma_20[-1])
    
    ema_5 = calculate_ema(closing_prices, 5)
    if len(ema_5) > 0:
        enhanced_s = np.append(enhanced_s, ema_5[-1])
    
    rsi = calculate_rsi(closing_prices, 14)
    enhanced_s = np.append(enhanced_s, rsi)
    
    macd = calculate_macd(closing_prices)
    if len(macd) > 0:
        enhanced_s = np.append(enhanced_s, macd[-1])
    
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20)
    enhanced_s = np.append(enhanced_s, upper_band[-1] if len(upper_band) > 0 else np.nan)
    enhanced_s = np.append(enhanced_s, lower_band[-1] if len(lower_band) > 0 else np.nan)
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    
    # Calculate daily returns
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    
    # Calculate historical volatility
    historical_vol = np.std(returns)
    
    # Calculate recent return
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    reward = 0
    
    # Relative threshold based on historical volatility
    threshold = 2 * historical_vol  # 2x historical volatility
    
    # Reward logic based on recent return and trend indicators
    if recent_return > threshold:
        reward += 50  # Strong positive trend
    elif recent_return < -threshold:
        reward -= 50  # Strong negative trend
    
    # Add more conditions based on additional features if needed
    # For example, using RSI value
    rsi_value = enhanced_s[-1]  # Last feature is RSI
    if rsi_value > 70:
        reward -= 20  # Overbought condition
    elif rsi_value < 30:
        reward += 20  # Oversold condition
    
    return reward