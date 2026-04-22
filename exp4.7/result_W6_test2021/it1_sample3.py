import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

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

def calculate_historical_volatility(prices):
    """Calculate historical volatility."""
    returns = np.diff(prices) / prices[:-1] * 100
    return np.std(returns)

def revise_state(s):
    closing_prices = s[0:20]
    volume = s[80:100]

    # Calculate features
    sma_5 = np.pad(calculate_sma(closing_prices, 5), (4, 0), 'constant', constant_values=np.nan)
    sma_10 = np.pad(calculate_sma(closing_prices, 10), (9, 0), 'constant', constant_values=np.nan)
    ema_5 = np.pad(calculate_ema(closing_prices, 5), (4, 0), 'constant', constant_values=np.nan)
    ema_10 = np.pad(calculate_ema(closing_prices, 10), (9, 0), 'constant', constant_values=np.nan)
    
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # Calculate historical volatility
    historical_volatility = calculate_historical_volatility(closing_prices)
    
    # Create enhanced state
    enhanced_s = np.concatenate((s, 
                                  sma_5, sma_10, 
                                  ema_5, ema_10, 
                                  np.array([historical_volatility]), 
                                  np.array([rsi_14])))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Percentage return
    historical_volatility = enhanced_s[120]  # Assuming it's the 121st feature

    # Define volatility-adaptive thresholds
    threshold = 2 * historical_volatility  # 2x std
    reward = 0

    # Evaluate the reward based on recent_return
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
    
    # Add conditions based on RSI
    rsi = enhanced_s[139]  # Assuming it's the 140th feature
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]