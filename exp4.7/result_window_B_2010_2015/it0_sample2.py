import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    return np.dot(weights, prices[-window:])

def calculate_rsi(prices, window=14):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    if loss == 0:
        return 100
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    if len(prices) < 26:
        return np.nan, np.nan
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    signal_line = calculate_ema(prices, 9)  # Signal line is the EMA of MACD
    return macd, signal_line

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume_data = s[80:99]
    
    enhanced_s = np.concatenate([s])
    
    # Technical indicators
    sma_5 = np.array([calculate_sma(closing_prices[:i+1], 5) for i in range(len(closing_prices))])
    sma_10 = np.array([calculate_sma(closing_prices[:i+1], 10) for i in range(len(closing_prices))])
    sma_20 = np.array([calculate_sma(closing_prices[:i+1], 20) for i in range(len(closing_prices))])
    
    ema_5 = np.array([calculate_ema(closing_prices[:i+1], 5) for i in range(len(closing_prices))])
    ema_10 = np.array([calculate_ema(closing_prices[:i+1], 10) for i in range(len(closing_prices))])
    
    rsi = np.array([calculate_rsi(closing_prices[:i+1]) for i in range(len(closing_prices))])
    
    macd, signal_line = zip(*[calculate_macd(closing_prices[:i+1]) for i in range(len(closing_prices))])
    
    # Append technical indicators to enhanced state
    enhanced_s = np.concatenate([enhanced_s, sma_5, sma_10, sma_20, ema_5, ema_10, rsi, macd, signal_line])
    
    # Handle edge cases for NaN values (replace with zeros or forward-fill)
    enhanced_s = np.nan_to_num(enhanced_s)
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate daily returns to find historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    threshold = 2 * historical_vol  # Adaptive threshold
    
    reward = 0
    
    # Determine reward based on recent return and trend indicators
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
    
    # Incorporate RSI as a measure of overbought/oversold
    rsi_value = enhanced_s[120:140][-1]  # Assuming RSI is appended at position 120 to 139
    if rsi_value < 30:
        reward += 20  # Oversold condition
    elif rsi_value > 70:
        reward -= 20  # Overbought condition
    
    # Limit the reward within the range [-100, 100]
    reward = max(-100, min(reward, 100))
    
    return reward