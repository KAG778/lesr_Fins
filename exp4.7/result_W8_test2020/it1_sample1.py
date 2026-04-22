import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Start with the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else np.mean(gain)
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else np.mean(loss)

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window=14):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-window:]) if len(tr) >= window else np.mean(tr)
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    highs = s[40:60]
    lows = s[60:80]
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(highs, lows, closing_prices, 14)
    
    # Create enhanced state
    enhanced_s = np.concatenate((
        s, 
        np.pad(sma_5, (4, 0), 'constant', constant_values=np.nan),
        np.pad(sma_10, (9, 0), 'constant', constant_values=np.nan),
        np.pad(ema_5, (4, 0), 'constant', constant_values=np.nan),
        np.pad(rsi_14, (13, 0), 'constant', constant_values=np.nan),
        np.pad(np.array([atr_14]), (13, 0), 'constant', constant_values=np.nan)
    ))

    return np.nan_to_num(enhanced_s)

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_volatility = np.std(returns)

    # Volatility-adaptive threshold
    threshold = 2 * historical_volatility  # Use 2x historical volatility

    reward = 0

    # Reward based on recent return
    if recent_return > threshold:  # Strong upward movement
        reward += 50
    elif recent_return < -threshold:  # Strong downward movement
        reward -= 50

    # Adjust reward based on RSI
    rsi = enhanced_s[-1]  # Assuming the last element is the latest RSI
    if rsi < 30:
        reward += 20  # Oversold condition, potential buy signal
    elif rsi > 70:
        reward -= 20  # Overbought condition, potential sell signal

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]