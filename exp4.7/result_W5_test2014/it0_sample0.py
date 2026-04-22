import numpy as np

def calculate_returns(prices):
    """ Calculate daily returns. """
    return np.diff(prices) / prices[:-1] * 100

def calculate_sma(prices, window):
    """ Calculate Simple Moving Average. """
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """ Calculate Exponential Moving Average. """
    return prices.ewm(span=window, adjust=False).mean()

def calculate_rsi(prices, window):
    """ Calculate Relative Strength Index. """
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]
    
    # Calculate additional features
    returns = calculate_returns(closing_prices)
    volatility = np.std(returns) if len(returns) > 0 else 0
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    rsi = calculate_rsi(closing_prices, 14)
    
    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        np.array([volatility]),
        sma_5[-1:],  # Last value of SMA 5
        sma_10[-1:], # Last value of SMA 10
        np.array([rsi])  # RSI value
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # Extract features from the enhanced state
    closing_prices = enhanced_s[0:20]
    recent_return = calculate_returns(closing_prices)[-1]  # Last day's return
    volatility = enhanced_s[120]  # Volatility feature
    rsi = enhanced_s[121]  # RSI feature

    # Calculate thresholds
    threshold = 2 * volatility  # Volatility-adaptive threshold

    # Initialize reward
    reward = 0

    # Positive reward for suitable trading conditions
    if recent_return > threshold:
        reward += 50  # Strong upward momentum

    # Penalize for downward moves
    if recent_return < -threshold:
        reward -= 50  # Significant downward move

    # Incorporate RSI into the reward structure
    if rsi < 30:
        reward += 20  # Oversold condition, potential buy signal
    elif rsi > 70:
        reward -= 20  # Overbought condition, potential sell signal

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]