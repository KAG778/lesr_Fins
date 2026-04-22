import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    ema = np.zeros(len(prices))
    ema[:window] = np.nan  # first 'window' values will be NaN
    
    # Initialize EMA with the first window's average
    ema[window-1] = np.mean(prices[:window])
    
    # Calculate the rest of the EMA values
    multiplier = 2 / (window + 1)
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    avg_gain[window-1] = np.mean(gain[:window])
    avg_loss[window-1] = np.mean(loss[:window])
    
    for i in range(window, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (window - 1) + gain[i-1]) / window
        avg_loss[i] = (avg_loss[i-1] * (window - 1) + loss[i-1]) / window
    
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

    enhanced_s = np.copy(s)

    # Calculate SMA, EMA, RSI (using a 14-day window for RSI)
    enhanced_s = np.concatenate((enhanced_s, 
                                  calculate_sma(closing_prices, 5),
                                  calculate_sma(closing_prices, 10),
                                  calculate_sma(closing_prices, 20),
                                  calculate_ema(closing_prices, 5),
                                  calculate_ema(closing_prices, 10),
                                  calculate_rsi(closing_prices, 14)))

    # Handle edge cases (e.g., NaN values for indicators)
    enhanced_s = np.nan_to_num(enhanced_s, nan=0.0)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)  # Daily volatility

    # Use a relative threshold
    threshold = 2 * historical_vol  # Adaptive threshold

    reward = 0

    # Reward logic
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return
    else:
        reward += 10  # Neutral reward for modest returns

    # Add additional criteria based on RSI or other indicators here
    rsi = enhanced_s[119]  # Assuming the last feature is RSI
    if rsi < 30:  # Oversold condition
        reward += 20
    elif rsi > 70:  # Overbought condition
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]