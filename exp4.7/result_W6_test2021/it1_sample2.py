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
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else np.mean(gain)
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else np.mean(loss)
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_volatility(prices, window):
    """Calculate historical volatility."""
    returns = np.diff(prices) / prices[:-1] * 100
    return np.std(returns)

def revise_state(s):
    # Extract price data
    closing_prices = s[0:20]

    # Calculate features
    sma_5 = np.concatenate((np.array([np.nan]*4), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.array([np.nan]*9), calculate_sma(closing_prices, 10)))
    sma_20 = np.concatenate((np.array([np.nan]*19), calculate_sma(closing_prices, 20)))
    
    ema_5 = np.concatenate((np.array([np.nan]*4), calculate_ema(closing_prices, 5)))
    ema_10 = np.concatenate((np.array([np.nan]*9), calculate_ema(closing_prices, 10)))
    
    rsi_14 = np.concatenate((np.array([np.nan]*13), np.array([calculate_rsi(closing_prices, 14)])))
    
    historical_volatility = np.zeros_like(closing_prices)
    for i in range(len(closing_prices)):
        if i >= 20:  # Calculate volatility only when enough data is available
            historical_volatility[i] = calculate_volatility(closing_prices[:i+1], 20)
        else:
            historical_volatility[i] = np.nan  # Not enough data to calculate
    
    # Construct the enhanced state
    enhanced_s = np.concatenate((s, 
                                  sma_5, sma_10, sma_20, 
                                  ema_5, ema_10, 
                                  rsi_14, 
                                  historical_volatility))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # Extract features
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    if len(closing_prices) > 1:
        recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    else:
        recent_return = 0

    # Calculate historical volatility
    historical_volatility = enhanced_s[139]  # Assuming historical volatility is the last feature

    # Use volatility-adaptive thresholds
    threshold = 2 * historical_volatility if historical_volatility > 0 else 1  # Avoid division by zero
    reward = 0

    # Reward logic based on recent return
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
    
    # Add conditions based on RSI
    rsi = enhanced_s[159]  # Assuming RSI is the second last feature
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]