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

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.abs(highs[-window:] - closes[-window:]), 
                    np.abs(lows[-window:] - closes[-window:]))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]

    # Create enhanced state with selected features
    enhanced_s = np.copy(s)

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Append features to enhanced state
    enhanced_s = np.concatenate((enhanced_s, 
                                  np.array([sma_5[-1] if sma_5.size > 0 else np.nan,
                                            sma_10[-1] if sma_10.size > 0 else np.nan,
                                            ema_5[-1] if ema_5.size > 0 else np.nan,
                                            rsi,
                                            macd[-1] if macd.size > 0 else np.nan,
                                            upper_band[-1] if upper_band.size > 0 else np.nan,
                                            lower_band[-1] if lower_band.size > 0 else np.nan,
                                            atr])))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    
    # Calculate daily returns and historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    
    # Calculate recent return
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    # Adaptive threshold based on historical volatility
    threshold = 2 * historical_vol  # 2x historical volatility
    reward = 0
    
    # Reward logic based on recent return
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement
    
    # Evaluate RSI for additional reward adjustments
    rsi = enhanced_s[-4]  # Assuming RSI is the fourth last feature
    if rsi < 30:
        reward += 20  # Potentially oversold
    elif rsi > 70:
        reward -= 20  # Potentially overbought
    
    return np.clip(reward, -100, 100)  # Ensure reward is within the range