import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Set the first EMA value to the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return np.mean(tr[-window:])

def revise_state(s):
    close_prices = s[0:20]
    open_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]
    
    # Adding technical indicators:
    sma_5 = np.concatenate((np.full(4, np.nan), calculate_sma(close_prices, 5)))
    sma_10 = np.concatenate((np.full(9, np.nan), calculate_sma(close_prices, 10)))
    ema_5 = calculate_ema(close_prices, 5)[-20:]
    rsi = np.concatenate((np.full(13, np.nan), calculate_rsi(close_prices, 14)))
    atr = np.concatenate((np.full(13, np.nan), calculate_atr(high_prices, low_prices, close_prices, 14)))

    # Enhanced state with additional features
    enhanced_s = np.concatenate((s, sma_5, sma_10, ema_5, rsi, atr))
    return enhanced_s

def intrinsic_reward(enhanced_s):
    close_prices = enhanced_s[0:20]
    recent_return = (close_prices[-1] - close_prices[-2]) / close_prices[-2] * 100  # Daily return in percentage
    rsi = enhanced_s[99]  # Latest RSI value
    atr = enhanced_s[118]  # Latest ATR value

    # Calculate historical volatility from closing prices
    returns = np.diff(close_prices) / close_prices[:-1] * 100
    historical_volatility = np.std(returns)
    
    # Relative thresholds for the reward function
    threshold = 2 * historical_volatility  # 2x historical volatility as threshold
    
    reward = 0
    
    # Reward/Penalty based on recent return and RSI
    if recent_return < -threshold:
        reward -= 50
    elif recent_return > threshold:
        reward += 50
        
    # Adjust reward based on RSI for additional trading context
    if rsi < 30:  # Oversold
        reward += 20
    elif rsi > 70:  # Overbought
        reward -= 20
    
    return np.clip(reward, -100, 100)  # Ensure the reward is within the range of [-100, 100]