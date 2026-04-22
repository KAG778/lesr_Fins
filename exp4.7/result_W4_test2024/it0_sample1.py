import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]  # First EMA value is the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]
    
    # Enhanced state
    enhanced_s = np.copy(s)
    
    # 5-day SMA
    sma_5 = calculate_sma(closing_prices, 5)
    enhanced_s = np.concatenate((enhanced_s, sma_5))

    # 10-day EMA
    ema_10 = calculate_ema(closing_prices, 10)
    enhanced_s = np.concatenate((enhanced_s, ema_10))

    # 14-day RSI
    rsi_14 = calculate_rsi(closing_prices, 14)
    enhanced_s = np.concatenate((enhanced_s, np.array([rsi_14])))

    # 20-day Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    volatility = np.std(returns) if len(returns) > 0 else 0
    enhanced_s = np.concatenate((enhanced_s, np.array([volatility])))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    volatility = enhanced_s[-1]  # Last feature is the calculated volatility

    # Set threshold based on historical volatility
    threshold = 2 * volatility  # Using 2x historical volatility

    # Initialize reward
    reward = 0

    # Trading conditions
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return
    
    # Using RSI for additional context (normalized to fit within [-100, 100])
    rsi = enhanced_s[-2]  # Second last feature is RSI
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition
    
    return np.clip(reward, -100, 100)  # Ensure reward stays within [-100, 100]