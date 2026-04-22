import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def calculate_roc(prices, window):
    return (prices[-1] - prices[-window - 1]) / prices[-window - 1] * 100

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)[-1:]  # Latest value of 5-day SMA
    sma_10 = calculate_sma(closing_prices, 10)[-1:]  # Latest value of 10-day SMA
    ema_5 = calculate_ema(closing_prices, 5)[-1:]  # Latest value of 5-day EMA
    ema_10 = calculate_ema(closing_prices, 10)[-1:]  # Latest value of 10-day EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # Latest value of RSI
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)  # Latest value of ATR
    roc_5 = calculate_roc(closing_prices, 5)  # Latest value of Rate of Change

    # Create enhanced state
    enhanced_s = np.concatenate((s, 
                                  sma_5, sma_10, 
                                  ema_5, ema_10, 
                                  np.array([rsi_14]), 
                                  np.array([atr_14]), 
                                  np.array([roc_5])))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    threshold = 2 * historical_volatility  # Adaptive threshold

    reward = 0

    # Reward logic based on recent return and thresholds
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return
    
    # Additional checks for risk management
    rsi = enhanced_s[-3]  # Last calculated RSI
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    # Penalize for high historical volatility
    if historical_volatility > np.mean(closing_prices) * 0.05:  # Adjusted threshold for high volatility
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]