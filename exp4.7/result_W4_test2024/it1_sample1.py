import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
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

def calculate_volatility(prices):
    """Calculate volatility as standard deviation of returns."""
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) if len(returns) > 0 else 0

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    enhanced_s = np.copy(s)

    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    volatility = calculate_volatility(closing_prices)
    
    # Concatenating the features (keeping the most effective ones)
    enhanced_s = np.concatenate((enhanced_s, 
                                  sma_5[-1:] if len(sma_5) > 0 else np.array([np.nan]),
                                  sma_10[-1:] if len(sma_10) > 0 else np.array([np.nan]),
                                  np.array([ema_10[-1]] if len(ema_10) > 0 else np.nan),
                                  np.array([rsi_14]),
                                  np.array([volatility])))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    volatility = enhanced_s[-1]  # Last feature is the calculated volatility

    # Set adaptive threshold based on historical volatility
    threshold = 2 * volatility  # Using 2x historical volatility

    reward = 0

    # Reward for recent return relative to threshold
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Reward for momentum based on RSI
    rsi = enhanced_s[-2]  # Second last feature is RSI
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    # Adding a mild reward for sideways market
    if -threshold <= recent_return <= threshold:
        reward += 10  # Mild return, encourage holding

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range