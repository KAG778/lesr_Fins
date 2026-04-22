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
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain[-window:]) > 0 else 0
    avg_loss = np.mean(loss[-window:]) if len(loss[-window:]) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    volumes = s[80:100]
    
    # Calculate various technical indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1]  # Latest 5-day SMA
    sma_10 = calculate_sma(closing_prices, 10)[-1]  # Latest 10-day SMA
    sma_20 = calculate_sma(closing_prices, 20)[-1]  # Latest 20-day SMA
    ema_5 = calculate_ema(closing_prices, 5)[-1]  # Latest 5-day EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # Latest 14-day RSI
    
    # Calculate daily returns and historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
    
    # Append new features to the state
    enhanced_s = np.concatenate((
        s,
        np.array([sma_5, sma_10, sma_20, ema_5, rsi_14, historical_volatility])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    historical_volatility = enhanced_s[-1]  # Last feature added is historical volatility
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Use 2x historical volatility as threshold
    threshold = 2 * historical_volatility if historical_volatility > 0 else 0
    
    reward = 0
    
    # Reward logic based on recent return and momentum
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
    
    # Add additional reward or penalty based on RSI
    rsi = enhanced_s[-2]  # Second last feature is RSI
    if rsi < 30:
        reward += 20  # Oversold condition, consider buying
    elif rsi > 70:
        reward -= 20  # Overbought condition, consider selling

    # Final reward should be clamped to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward