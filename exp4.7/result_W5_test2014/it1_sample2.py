import numpy as np

def calculate_returns(prices):
    """ Calculate daily returns. """
    return np.diff(prices) / prices[:-1] * 100

def calculate_sma(prices, window):
    """ Calculate Simple Moving Average (SMA). """
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """ Calculate Exponential Moving Average (EMA). """
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window - 1] = np.mean(prices[:window])  # First EMA is SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window=14):
    """ Calculate Relative Strength Index (RSI). """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """ Calculate Moving Average Convergence Divergence (MACD). """
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    return macd[-1] if len(macd) > 0 else np.nan

def calculate_volatility(prices):
    """ Calculate historical volatility. """
    returns = calculate_returns(prices)
    return np.std(returns) if len(returns) > 0 else 0

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]
    
    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    rsi = calculate_rsi(closing_prices)
    macd = calculate_macd(closing_prices)
    volatility = calculate_volatility(closing_prices)

    # Create enhanced state
    enhanced_s = np.concatenate((s, 
                                  [sma_5, sma_10, ema_5, rsi, macd, volatility, volume[-1]]))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = calculate_returns(closing_prices)[-1]  # Last day's return
    volatility = enhanced_s[-1]  # Volatility feature

    # Calculate thresholds
    threshold = 2 * volatility  # Volatility-adaptive threshold

    # Initialize reward
    reward = 0

    # Positive reward for suitable trading conditions
    if recent_return > threshold:
        reward += 50  # Strong upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Significant downward move

    # Incorporate RSI into the reward structure
    rsi = enhanced_s[4]  # RSI feature
    if rsi < 30:
        reward += 20  # Oversold condition, potential buy signal
    elif rsi > 70:
        reward -= 20  # Overbought condition, potential sell signal

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]