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
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    return np.dot(weights, prices[-window:])

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = (deltas[deltas > 0].sum() / window) if window > 0 else 0
    loss = (-deltas[deltas < 0].sum() / window) if window > 0 else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    return ema_12 - ema_26

def calculate_bollinger_bands(prices, window):
    """Calculate Bollinger Bands"""
    if len(prices) < window:
        return np.nan, np.nan
    sma = calculate_sma(prices, window)
    std_dev = np.std(prices[-window:])
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    return upper_band, lower_band

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:59]
    low_prices = s[60:79]

    # Create enhanced state with selected features
    enhanced_s = np.concatenate([
        closing_prices,
        np.array([
            calculate_sma(closing_prices, 5),
            calculate_sma(closing_prices, 10),
            calculate_ema(closing_prices, 5),
            calculate_ema(closing_prices, 10),
            calculate_rsi(closing_prices, 14),
            *calculate_bollinger_bands(closing_prices, 20),  # Upper and Lower Bands
            calculate_macd(closing_prices)  # MACD
        ])
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    
    # Use 2x historical volatility as adaptive threshold
    threshold = 2 * historical_vol
    
    reward = 0
    
    # Reward based on recent return
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement
    
    # Include RSI in the reward calculation
    rsi = enhanced_s[5]  # Assuming RSI is at index 5 in enhanced state
    if rsi < 30:
        reward += 20  # Potentially oversold
    elif rsi > 70:
        reward -= 20  # Potentially overbought
    
    return np.clip(reward, -100, 100)  # Ensure reward is within the range