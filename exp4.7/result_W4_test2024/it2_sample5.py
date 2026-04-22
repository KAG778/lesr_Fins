import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window+1:]:
        ema = (price * alpha) + (ema * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_volatility(prices):
    """Calculate historical volatility as standard deviation of returns."""
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) if len(returns) > 0 else 0

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    enhanced_s = np.copy(s)

    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    volatility = calculate_volatility(closing_prices)
    
    # Append features to the enhanced state
    enhanced_s = np.concatenate((enhanced_s, 
                                  [sma_5], 
                                  [sma_10], 
                                  [ema_10], 
                                  [rsi_14], 
                                  [volatility]))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = enhanced_s[-1]  # Last feature is historical volatility

    # Volatility-adaptive threshold
    threshold = 2 * historical_vol
    reward = 0

    # Reward logic based on recent return
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Additional checks for trends using moving averages
    sma_5 = enhanced_s[-5]  # Last feature added
    sma_10 = enhanced_s[-4]  # Second last feature added
    if sma_5 > sma_10:  # Bullish trend
        reward += 20
    elif sma_5 < sma_10:  # Bearish trend
        reward -= 20

    # Encourage holding during mild returns
    if -threshold <= recent_return <= threshold:
        reward += 10  # Mild return, encourage holding

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]