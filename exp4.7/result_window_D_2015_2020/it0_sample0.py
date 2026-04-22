import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
    for t in range(1, len(prices)):
        ema[t] = (prices[t] * alpha) + (ema[t - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = calculate_sma(gain, window)
    avg_loss = calculate_sma(loss, window)
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_window=12, long_window=26):
    ema_short = calculate_ema(prices, short_window)
    ema_long = calculate_ema(prices, long_window)
    macd = ema_short[long_window-1:] - ema_long[long_window-1:]
    return macd

def calculate_atr(highs, lows, closes, window):
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = calculate_sma(true_ranges, window)
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]

    # Calculate additional features
    sma_5 = np.concatenate((np.array([np.nan]*4), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.array([np.nan]*9), calculate_sma(closing_prices, 10)))
    sma_20 = np.concatenate((np.array([np.nan]*19), calculate_sma(closing_prices, 20)))
    
    ema_5 = np.concatenate((np.array([np.nan]*4), calculate_ema(closing_prices, 5)))
    ema_10 = np.concatenate((np.array([np.nan]*9), calculate_ema(closing_prices, 10)))
    
    rsi = np.concatenate((np.array([np.nan]*13), calculate_rsi(closing_prices, 14)))
    
    macd = np.concatenate((np.array([np.nan]*25), calculate_macd(closing_prices)))
    
    atr = np.concatenate((np.array([np.nan]*19), calculate_atr(high_prices, low_prices, closing_prices, 14)))

    # Combine original state with new features
    enhanced_s = np.concatenate([s, sma_5, sma_10, sma_20, ema_5, ema_10, rsi, macd, atr])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Recent daily return

    # Calculate historical volatility from closing prices
    historical_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(historical_returns)  # Daily volatility (in percentage)

    # Define threshold based on historical volatility
    threshold = 2 * historical_vol

    reward = 0

    # Evaluate the state based on the recent return and volatility
    if recent_return > threshold:
        reward += 50  # Positive reward for strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Negative reward for strong downward movement

    # Add additional conditions based on RSI
    rsi = enhanced_s[100:120]  # Assuming RSI starts from index 100
    if rsi[-1] < 30:
        reward += 20  # Oversold condition
    elif rsi[-1] > 70:
        reward -= 20  # Overbought condition

    # Cap the reward to the defined range of [-100, 100]
    reward = max(min(reward, 100), -100)

    return reward