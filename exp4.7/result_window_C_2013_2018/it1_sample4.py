import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    ema = np.zeros(len(prices))
    ema[window-1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * (2 / (window + 1)) + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) > window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) > window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_historical_volatility(returns):
    """Calculate historical volatility based on returns"""
    return np.std(returns) if len(returns) > 0 else 0

def revise_state(s):
    closing_prices = s[0:20]
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1]  # Latest SMA
    sma_10 = calculate_sma(closing_prices, 10)[-1]  # Latest SMA
    ema_5 = calculate_ema(closing_prices, 5)[-1]  # Latest EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # Latest RSI
    
    # Calculate daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_volatility = calculate_historical_volatility(daily_returns)

    # Create enhanced state
    enhanced_s = np.concatenate((
        s, 
        np.array([sma_5, sma_10, ema_5, rsi_14, historical_volatility])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = enhanced_s[-1]  # Last feature is historical volatility

    # Use adaptive thresholds based on historical volatility
    threshold = 2 * historical_volatility if historical_volatility > 0 else 0

    reward = 0

    # Reward logic based on recent return and momentum
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum

    # Add adjustments based on RSI
    rsi = enhanced_s[-2]
    if rsi < 30:
        reward += 20  # Oversold condition, consider buying
    elif rsi > 70:
        reward -= 20  # Overbought condition, consider selling

    # Ensure the reward is clamped within the specified range
    return np.clip(reward, -100, 100)