import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * (prices[i] - ema[i-1]) + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_volatility(prices):
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) if len(returns) > 0 else 0

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    enhanced_s = np.copy(s)

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    volatility = calculate_volatility(closing_prices)
    
    # Append features to the enhanced state
    enhanced_s = np.concatenate((enhanced_s, 
                                  np.array([sma_5[-1] if len(sma_5) > 0 else np.nan]), 
                                  np.array([sma_10[-1] if len(sma_10) > 0 else np.nan]), 
                                  np.array([ema_10[-1]]), 
                                  np.array([rsi_14]), 
                                  np.array([volatility])))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = enhanced_s[-1]  # Last feature is the calculated volatility

    # Volatility-adaptive threshold
    threshold = 2 * historical_vol
    reward = 0

    # Reward for positive momentum (recent return)
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Additional checks for trends using moving averages
    sma_5 = enhanced_s[-3]  # Last 3rd feature is SMA_5
    sma_10 = enhanced_s[-2]  # Last 2nd feature is SMA_10
    if sma_5 > sma_10:  # Bullish trend
        reward += 20
    elif sma_5 < sma_10:  # Bearish trend
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range