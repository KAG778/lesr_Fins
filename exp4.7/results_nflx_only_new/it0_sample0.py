import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window or window <= 0:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window or window <= 0:
        return np.nan
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    return np.dot(weights, prices[-window:])

def calculate_rsi(prices, window):
    if len(prices) < window or window <= 0:
        return np.nan
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, short_window=12, long_window=26):
    if len(prices) < long_window:
        return np.nan, np.nan
    ema_short = calculate_ema(prices, short_window)
    ema_long = calculate_ema(prices, long_window)
    macd = ema_short - ema_long
    signal_line = calculate_ema([macd] * short_window, short_window)
    return macd, signal_line

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    
    enhanced_s = np.zeros(120 + 8)  # Original + new features (8 technical indicators)
    
    # Copy original state
    enhanced_s[0:120] = s
    
    # Calculate additional features
    enhanced_s[120] = calculate_sma(closing_prices, 5)
    enhanced_s[121] = calculate_sma(closing_prices, 10)
    enhanced_s[122] = calculate_sma(closing_prices, 20)
    enhanced_s[123] = calculate_ema(closing_prices, 5)
    enhanced_s[124] = calculate_ema(closing_prices, 10)
    enhanced_s[125] = calculate_rsi(closing_prices, 14)
    macd, signal_line = calculate_macd(closing_prices)
    enhanced_s[126] = macd
    enhanced_s[127] = signal_line
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Define reward
    reward = 0
    
    # Reward based on recent return relative to historical volatility
    threshold = 2 * historical_volatility if historical_volatility > 0 else 0
    
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return > 0:
        reward += 20  # Moderate positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return
    elif recent_return < 0:
        reward -= 20  # Moderate negative return
    
    # Additional reward for trend and risk considerations can be added here
    
    return reward