import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Initialize EMA with the first price
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    """Calculate MACD and Signal Line."""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12[11:] - ema_26[25:]  # Align lengths
    signal_line = calculate_ema(macd, 9)
    return macd, signal_line

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:119]
    
    enhanced_s = np.copy(s)
    
    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi = calculate_rsi(closing_prices, 14)
    macd, signal_line = calculate_macd(closing_prices)

    # Append calculated indicators to the enhanced state
    enhanced_s = np.concatenate((enhanced_s, np.concatenate((sma_5[-1:], sma_10[-1:], sma_20[-1:], 
                                                              ema_5[-1:], ema_10[-1:], 
                                                              [rsi], macd[-1:], signal_line[-1:]))))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    
    # Set thresholds relative to volatility
    threshold = 2 * historical_volatility
    
    reward = 0
    
    # Determine reward based on recent return and RSI
    rsi = enhanced_s[-1]  # Assuming last value is RSI from the enhanced state
    
    if recent_return > threshold:
        reward += 50  # Positive trade signal
    elif recent_return < -threshold:
        reward -= 50  # Negative trade signal
    
    # Adjust based on RSI
    if rsi > 70:
        reward -= 25  # Overbought condition
    elif rsi < 30:
        reward += 25  # Oversold condition
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]