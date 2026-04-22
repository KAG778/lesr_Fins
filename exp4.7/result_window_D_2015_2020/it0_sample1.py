import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = -np.where(deltas < 0, deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    if avg_loss == 0:
        return 100  # Prevent division by zero
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]
    adjusted_closing_prices = s[100:119]
    
    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # Pad the arrays to maintain dimensions for the original state
    sma_5 = np.pad(sma_5, (4, 0), 'constant', constant_values=np.nan)
    sma_10 = np.pad(sma_10, (9, 0), 'constant', constant_values=np.nan)
    sma_20 = np.pad(sma_20, (19, 0), 'constant', constant_values=np.nan)
    ema_5 = np.pad(ema_5, (4, 0), 'constant', constant_values=np.nan)
    ema_10 = np.pad(ema_10, (9, 0), 'constant', constant_values=np.nan)

    # Enhanced state: concatenate original state with new features
    enhanced_s = np.concatenate((
        s, 
        sma_5, sma_10, sma_20, 
        ema_5, ema_10, 
        np.full((1,), rsi_14)  # RSI is a single value
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    
    # Define threshold based on historical volatility
    threshold = 2 * historical_vol
    
    reward = 0
    # Reward calculation based on conditions
    if recent_return > threshold:
        reward += 50
    elif recent_return < -threshold:
        reward -= 50

    # Add additional reward criteria (e.g., based on RSI)
    rsi_value = enhanced_s[-1]  # Last value is the RSI
    if rsi_value < 30:
        reward += 25  # Potential buy signal
    elif rsi_value > 70:
        reward -= 25  # Potential sell signal

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]