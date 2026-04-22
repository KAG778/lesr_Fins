import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs, lows, closes, window):
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.mean(true_ranges[-window:])  # Return the average true range over the window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Calculate new features
    sma_5 = np.concatenate((np.array([np.nan]*4), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.array([np.nan]*9), calculate_sma(closing_prices, 10)))
    sma_20 = np.concatenate((np.array([np.nan]*19), calculate_sma(closing_prices, 20)))
    
    ema_5 = np.concatenate((np.array([np.nan]*4), calculate_ema(closing_prices, 5)))
    ema_10 = np.concatenate((np.array([np.nan]*9), calculate_ema(closing_prices, 10)))
    
    rsi = np.concatenate((np.array([np.nan]*13), np.array([calculate_rsi(closing_prices, 14)])))
    atr = np.concatenate((np.array([np.nan]*13), np.array([calculate_atr(high_prices, low_prices, closing_prices, 14)])))
    
    # Combine original state with new features
    enhanced_s = np.concatenate([s, sma_5, sma_10, sma_20, ema_5, ema_10, rsi, atr])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Recent daily return

    # Calculate historical volatility from closing prices
    historical_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(historical_returns)  # Daily volatility (in percentage)

    # Define thresholds based on historical volatility
    threshold = 2 * historical_vol

    reward = 0

    # Reward calculation based on recent return
    if recent_return > threshold:
        reward += 50  # Positive reward for strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Negative reward for strong downward movement

    # Add additional conditions based on RSI
    rsi = enhanced_s[-2]  # Assuming RSI is the second to last feature
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    # Cap the reward to the defined range of [-100, 100]
    return np.clip(reward, -100, 100)