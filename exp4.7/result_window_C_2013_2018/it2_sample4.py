import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]  # Start with the first value
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
    if loss == 0:
        return 100  # To handle division by zero
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_historical_volatility(returns):
    """Calculate historical volatility based on returns"""
    return np.std(returns) if len(returns) > 0 else 0

def revise_state(s):
    closing_prices = s[0:20]
    volumes = s[80:100]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)  # Latest 5-day SMA
    sma_10 = calculate_sma(closing_prices, 10)  # Latest 10-day SMA
    ema_5 = calculate_ema(closing_prices, 5)  # Latest 5-day EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # Latest 14-day RSI

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
    historical_volatility = enhanced_s[-1]  # Last feature added is historical volatility

    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Use volatility-adaptive thresholds
    threshold = 2 * historical_volatility if historical_volatility > 0 else 0

    reward = 0

    # Reward logic based on recent return and momentum
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum

    # Adjust reward based on RSI
    rsi = enhanced_s[-2]
    if rsi < 30:
        reward += 20  # Oversold condition, consider buying
    elif rsi > 70:
        reward -= 20  # Overbought condition, consider selling

    # Ensure the reward is clamped within the specified range
    return np.clip(reward, -100, 100)