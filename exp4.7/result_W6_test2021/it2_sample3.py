import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else np.mean(gain)
    avg_loss = np.mean(loss[-window:] if len(loss) >= window else np.mean(loss))
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_volatility(prices):
    """Calculate historical volatility."""
    returns = np.diff(prices) / prices[:-1] * 100
    return np.std(returns)

def revise_state(s):
    """Revise state representation by adding technical indicators."""
    closing_prices = s[0:20]
    
    # Calculate moving averages
    sma_5 = np.pad(calculate_sma(closing_prices, 5), (4, 0), 'edge')
    sma_10 = np.pad(calculate_sma(closing_prices, 10), (9, 0), 'edge')
    ema_5 = np.pad(calculate_ema(closing_prices, 5), (4, 0), 'edge')
    ema_10 = np.pad(calculate_ema(closing_prices, 10), (9, 0), 'edge')
    
    rsi_14 = np.pad(np.array([calculate_rsi(closing_prices, 14)]), (13, 0), 'edge')
    
    # Calculate historical volatility
    historical_volatility = calculate_volatility(closing_prices)
    
    # Create enhanced state
    enhanced_s = np.concatenate((
        s, 
        sma_5, sma_10, 
        ema_5, ema_10, 
        np.array([historical_volatility]), 
        rsi_14
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    """Calculate intrinsic reward based on recent return and RSI."""
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) > 1 else 0
    historical_volatility = enhanced_s[-1]  # Assuming it's the last feature

    # Use volatility-adaptive thresholds
    threshold = 2 * historical_volatility if historical_volatility > 0 else 1  # Avoid division by zero
    reward = 0

    # Reward logic based on recent return
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
    
    # Add conditions based on RSI
    rsi = enhanced_s[-2]  # Assuming RSI is the second last feature
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]