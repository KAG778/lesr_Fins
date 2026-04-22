import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]  # Start with the first value
    for price in prices[-window + 1:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    if len(prices) < window + 1:
        return np.nan
    deltas = np.diff(prices[-(window + 1):])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    if loss == 0:
        return 100  # To handle division by zero
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]
    
    # Calculate Technical Indicators
    sma_5 = calculate_sma(closing_prices, 5)  # 5-day SMA
    sma_10 = calculate_sma(closing_prices, 10)  # 10-day SMA
    ema_5 = calculate_ema(closing_prices, 5)  # 5-day EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # 14-day RSI
    
    # Calculate daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    
    # Calculate historical volatility
    historical_volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
    
    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        np.array([sma_5, sma_10, ema_5, rsi_14, historical_volatility])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = enhanced_s[-1]  # Last feature added is historical volatility
    
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