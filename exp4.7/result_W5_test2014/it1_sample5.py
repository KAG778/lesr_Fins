import numpy as np

def calculate_returns(prices):
    """ Calculate daily returns. """
    return np.diff(prices) / prices[:-1] * 100

def calculate_sma(prices, window):
    """ Calculate Simple Moving Average. """
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """ Calculate Exponential Moving Average. """
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window - 1] = np.mean(prices[:window])  # First EMA is SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window=14):
    """ Calculate Relative Strength Index. """
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    if avg_loss == 0:
        return 100  # Avoid division by zero
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:59]
    low_prices = s[60:79]
    
    # Calculate additional features
    returns = calculate_returns(closing_prices)
    volatility = np.std(returns) if len(returns) > 0 else 0

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices, 14)
    
    # Calculate Average True Range (ATR) for risk management
    atr = np.mean(np.maximum(high_prices[-14:] - low_prices[-14:], 
                             np.maximum(np.abs(high_prices[-14:] - closing_prices[-14:]), 
                                        np.abs(low_prices[-14:] - closing_prices[-14:]))))

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        np.array([volatility]),
        sma_5[-1:],  # Last value of SMA 5
        sma_10[-1:], # Last value of SMA 10
        np.array([rsi]),  # RSI value
        np.array([ema_5[-1]]),  # Last value of EMA 5
        np.array([atr])  # Average True Range
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = calculate_returns(closing_prices)[-1]  # Last day's return
    volatility = enhanced_s[120]  # Volatility feature
    rsi = enhanced_s[121]  # RSI feature
    atr = enhanced_s[122]  # Average True Range
    
    # Calculate thresholds based on volatility
    threshold = 2 * volatility  # Volatility-adaptive threshold

    # Initialize reward
    reward = 0

    # Positive reward for suitable trading conditions
    if recent_return > threshold:
        reward += 50  # Strong upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Significant downward move
    
    # Incorporate RSI into the reward structure
    if rsi < 30:
        reward += 20  # Oversold condition, potential buy signal
    elif rsi > 70:
        reward -= 20  # Overbought condition, potential sell signal

    # Risk management based on ATR
    if np.abs(recent_return) > atr:
        reward -= 30  # Penalize high risk

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]