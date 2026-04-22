import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if window < len(gain) else np.mean(gain)
    avg_loss = np.mean(loss[-window:]) if window < len(loss) else np.mean(loss)
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return np.array([rsi] * (window - 1) + [rsi])  # Pad to match length

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:119]
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # Padding SMA and EMA to maintain array size
    sma_5 = np.pad(sma_5, (4, 0), 'edge')
    sma_10 = np.pad(sma_10, (9, 0), 'edge')
    sma_20 = np.pad(sma_20, (19, 0), 'edge')
    
    ema_5 = np.pad(ema_5, (4, 0), 'edge')
    ema_10 = np.pad(ema_10, (9, 0), 'edge')
    
    # Adding all features together
    enhanced_s = np.concatenate((s, sma_5, sma_10, sma_20, ema_5, ema_10, rsi_14))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical returns to determine volatility
    historical_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(historical_returns)
    
    # Determine thresholds
    threshold = 2 * historical_vol
    
    # Initialize reward
    reward = 0
    
    # Reward logic based on recent return and trend indicators (using RSI)
    rsi = enhanced_s[119]  # Last RSI value
    if recent_return > threshold and rsi < 70:  # Uptrend with RSI not overbought
        reward += 50
    elif recent_return < -threshold and rsi > 30:  # Downtrend with RSI not oversold
        reward -= 50
    
    return np.clip(reward, -100, 100)