import numpy as np

def calculate_returns(prices):
    """ Calculate daily returns. """
    return np.diff(prices) / prices[:-1] * 100

def calculate_sma(prices, window):
    """ Calculate Simple Moving Average. """
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """ Calculate Exponential Moving Average. """
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window - 1] = np.mean(prices[:window])  # First EMA is SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window=14):
    """ Calculate Relative Strength Index. """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    if avg_loss == 0:
        return 100  # Avoid division by zero
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """ Calculate MACD and Signal Line. """
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    signal_line = calculate_ema(macd[-9:], 9)  # Signal line is 9-day EMA of MACD
    return macd[-1], signal_line  # Return the last values

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]

    # Calculate technical indicators
    returns = calculate_returns(closing_prices)
    volatility = np.std(returns) if len(returns) > 0 else 0
    sma_5 = calculate_sma(closing_prices, 5)[-1]  # Last value of SMA 5
    sma_10 = calculate_sma(closing_prices, 10)[-1]  # Last value of SMA 10
    rsi = calculate_rsi(closing_prices, 14)  # RSI value
    macd, signal_line = calculate_macd(closing_prices)

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        np.array([volatility]),
        np.array([sma_5, sma_10, rsi, macd, signal_line])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = calculate_returns(closing_prices)[-1]  # Last day's return
    volatility = enhanced_s[120]  # Volatility feature

    # Use volatility-adaptive thresholds
    threshold = 2 * volatility  # 2x historical volatility threshold

    # Initialize reward
    reward = 0

    # Reward structure based on recent return and trend indication
    if recent_return > threshold:
        reward += 50  # Strong upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Significant downward move

    # Incorporate RSI into the reward structure
    rsi = enhanced_s[123]  # RSI feature
    macd = enhanced_s[124]  # MACD value
    signal_line = enhanced_s[125]  # Signal line

    if rsi < 30:
        reward += 20  # Oversold condition, potential buy signal
    elif rsi > 70:
        reward -= 20  # Overbought condition, potential sell signal

    # Trend confirmation via MACD
    if macd > signal_line:
        reward += 10  # Bullish signal
    elif macd < signal_line:
        reward -= 10  # Bearish signal

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]