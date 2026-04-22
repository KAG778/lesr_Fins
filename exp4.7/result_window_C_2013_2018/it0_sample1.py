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
    
    enhanced_s = np.zeros(140)  # New state with additional features

    # Original prices
    enhanced_s[0:120] = s

    # Calculate Technical Indicators
    enhanced_s[120] = calculate_sma(closing_prices, 5)  # 5-day SMA
    enhanced_s[121] = calculate_sma(closing_prices, 10)  # 10-day SMA
    enhanced_s[122] = calculate_ema(closing_prices, 10)  # 10-day EMA
    enhanced_s[123] = calculate_rsi(closing_prices, 14)  # 14-day RSI

    # Calculate daily returns and volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    if len(daily_returns) > 0:
        historical_volatility = np.std(daily_returns)
    else:
        historical_volatility = 0.0

    enhanced_s[124] = historical_volatility  # Volatility

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = enhanced_s[124]

    # Determine reward based on recent return relative to historical volatility
    if historical_volatility > 0:
        threshold = 2 * historical_volatility
    else:
        threshold = 0

    reward = 0

    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum

    # Reward adjustments based on RSI
    rsi = enhanced_s[123]
    if rsi < 30:
        reward += 20  # Potential buy signal
    elif rsi > 70:
        reward -= 20  # Potential sell signal

    return np.clip(reward, -100, 100)  # Clipping reward to range [-100, 100]