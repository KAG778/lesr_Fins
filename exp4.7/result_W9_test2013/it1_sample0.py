import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])  # Return the mean of the last 'window'

def revise_state(s):
    # Extract price and volume data from the raw state
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)[-1:]  # Last value of 5-day SMA
    sma_10 = calculate_sma(closing_prices, 10)[-1:]  # Last value of 10-day SMA
    ema_10 = calculate_ema(closing_prices, 10)[-1:]  # Last value of 10-day EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # Last value of 14-day RSI
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)  # Last value of 14-day ATR

    # Create enhanced state
    enhanced_s = np.concatenate((s,
                                  sma_5, sma_10, ema_10,
                                  np.array([rsi_14]),
                                  np.array([atr_14])))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    
    # Calculate daily returns
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    
    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    
    # Calculate recent return
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    # Define reward
    reward = 0

    # Use adaptive thresholds for rewards
    positive_threshold = 2 * historical_vol
    negative_threshold = -2 * historical_vol

    # Reward calculation based on recent return and thresholds
    if recent_return > positive_threshold:
        reward += 50  # Positive momentum
    elif recent_return < negative_threshold:
        reward -= 50  # Negative momentum
    
    # Include additional checks for risk management
    rsi = enhanced_s[-2]  # Last calculated RSI
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within specified range