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
    ema = prices[-window]
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

def calculate_historical_volatility(prices):
    """Calculate Historical Volatility"""
    if len(prices) < 2:
        return 0.0
    daily_returns = np.diff(prices) / prices[:-1] * 100
    return np.std(daily_returns) if len(daily_returns) > 0 else 0.0

def revise_state(s):
    closing_prices = s[0:20]
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)  # 5-day SMA
    sma_10 = calculate_sma(closing_prices, 10)  # 10-day SMA
    ema_10 = calculate_ema(closing_prices, 10)  # 10-day EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # 14-day RSI
    historical_volatility = calculate_historical_volatility(closing_prices)  # Historical volatility

    # Create enhanced state with effective features
    enhanced_s = np.concatenate((
        s, 
        np.array([sma_5, sma_10, ema_10, rsi_14, historical_volatility])
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
        reward += 50  # Strong upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong downward momentum

    # Adjust reward based on RSI
    rsi = enhanced_s[-2]  # Second last feature is RSI
    if rsi < 30:
        reward += 20  # Oversold condition, consider buying
    elif rsi > 70:
        reward -= 20  # Overbought condition, consider selling

    # Ensure the reward is clamped within the specified range
    return np.clip(reward, -100, 100)