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
    atr = np.mean(true_ranges[-window:])  # Average true range
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]

    # Calculate new features
    sma_5 = np.concatenate((np.array([np.nan]*4), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.array([np.nan]*9), calculate_sma(closing_prices, 10)))
    ema_5 = np.concatenate((np.array([np.nan]*4), calculate_ema(closing_prices, 5)))
    ema_10 = np.concatenate((np.array([np.nan]*9), calculate_ema(closing_prices, 10)))
    rsi = np.concatenate((np.array([np.nan]*13), [calculate_rsi(closing_prices, 14)]))  # Single value at the end
    atr = np.concatenate((np.array([np.nan]*13), [calculate_atr(high_prices, lows, closing_prices, 14)]))  # Single value at the end

    # Combine original state with new features
    enhanced_s = np.concatenate([s, sma_5, sma_10, ema_5, ema_10, rsi, atr])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return percentage
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if returns.size > 0 else 0
    
    # Define adaptive thresholds based on volatility
    threshold_up = 2 * historical_vol
    threshold_down = -2 * historical_vol
    
    reward = 0
    
    # Reward adjustments based on recent return
    if recent_return > threshold_up:
        reward += 50  # Positive reward for high gain
    elif recent_return < threshold_down:
        reward -= 50  # Negative reward for high loss
        
    # Assess RSI for overbought/oversold conditions
    rsi = enhanced_s[-2]  # Assuming RSI is at the second to last feature
    if rsi < 30:
        reward += 25  # Oversold condition
    elif rsi > 70:
        reward -= 25  # Overbought condition

    # Ensure reward is within the range [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward