import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window - 1] = np.mean(prices[:window])  # First EMA is SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    if avg_loss == 0:
        return 100  # Avoid division by zero, RSI is at its max
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def revise_state(s):
    # Extracting relevant price data
    close_prices = s[0:20]
    open_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]
    
    # Calculate additional technical indicators
    sma_5 = calculate_sma(close_prices, 5)
    sma_10 = calculate_sma(close_prices, 10)
    sma_20 = calculate_sma(close_prices, 20)

    ema_5 = calculate_ema(close_prices, 5)
    ema_10 = calculate_ema(close_prices, 10)
    
    rsi = np.array([calculate_rsi(close_prices[i:i+14], 14) for i in range(len(close_prices) - 14 + 1)])  # RSI for last 14 days
    
    # Pad the resulting arrays to maintain the dimension
    sma_5 = np.pad(sma_5, (4, 0), 'edge')  # 4 edge values to maintain length
    sma_10 = np.pad(sma_10, (9, 0), 'edge')  # 9 edge values to maintain length
    sma_20 = np.pad(sma_20, (19, 0), 'edge')  # 19 edge values to maintain length

    ema_5 = np.pad(ema_5, (4, 0), 'edge')
    ema_10 = np.pad(ema_10, (9, 0), 'edge')
    
    rsi = np.pad(rsi, (13, 0), 'edge')  # 13 edge values for the RSI

    # Combine all features into the enhanced state
    enhanced_s = np.concatenate((s, sma_5[:20], sma_10[:20], sma_20[:20], ema_5[:20], ema_10[:20], rsi[:20]))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # Extract closing prices for reward calculation
    close_prices = enhanced_s[0:20]
    
    # Calculate daily returns
    returns = np.diff(close_prices) / close_prices[:-1] * 100
    recent_return = returns[-1]  # Last return
    
    # Calculate historical volatility
    historical_volatility = np.std(returns)
    
    # Define reward initialization
    reward = 0
    
    # Use volatility-adaptive thresholds
    threshold = 2 * historical_volatility  # 2x historical volatility threshold
    
    # Adjust reward based on recent return
    if recent_return < -threshold:
        reward -= 50  # Strong negative return
    elif recent_return > threshold:
        reward += 50  # Strong positive return
    else:
        reward += 10  # Neutral or positive return

    # Example condition for trend indication (using SMA)
    sma_5 = enhanced_s[120:140]  # SMA 5
    sma_10 = enhanced_s[140:160]  # SMA 10
    if sma_5[-1] > sma_10[-1]:
        reward += 20  # Indicates a potential uptrend
    elif sma_5[-1] < sma_10[-1]:
        reward -= 20  # Indicates a potential downtrend

    return np.clip(reward, -100, 100)  # Ensure the reward is within the specified range