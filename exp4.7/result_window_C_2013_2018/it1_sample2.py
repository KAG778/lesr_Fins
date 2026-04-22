import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window)/window, mode='valid')[-1]  # Return the latest value

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    if len(prices) < window:
        return np.nan
    ema = prices[-window]  # Start with the first value
    alpha = 2 / (window + 1)
    for price in prices[-window + 1:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    if len(prices) < window + 1:
        return np.nan
    deltas = np.diff(prices[-(window + 1):])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_volatility(prices):
    """Calculate Historical Volatility"""
    if len(prices) < 2:
        return 0.0
    daily_returns = np.diff(prices) / prices[:-1] * 100
    return np.std(daily_returns) if len(daily_returns) > 0 else 0.0

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:119]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)   # 5-day SMA
    ema_10 = calculate_ema(closing_prices, 10)  # 10-day EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # 14-day RSI
    historical_volatility = calculate_volatility(closing_prices)  # Historical volatility

    # Create enhanced state with effective features
    enhanced_s = np.concatenate((
        s, 
        np.array([sma_5, ema_10, rsi_14, historical_volatility])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = enhanced_s[-1]  # Last feature is historical volatility

    # Use volatility-adaptive thresholds
    threshold = 2 * historical_volatility
    
    reward = 0

    # Reward logic based on recent return and momentum
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Adjust reward based on RSI
    rsi = enhanced_s[-2]  # Second last feature is RSI
    if rsi < 30:
        reward += 20  # Oversold condition, consider buying
    elif rsi > 70:
        reward -= 20  # Overbought condition, consider selling

    # Clipping reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward