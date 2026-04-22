import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema[-1]

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.maximum(np.abs(highs[-window:] - closes[-window:]), 
                               np.abs(lows[-window:] - closes[-window:])))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volume = s[80:99]
    adjusted_closing_prices = s[100:119]
    
    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # Add new features to the state
    enhanced_s = np.concatenate((s, 
                                  np.array([sma_5, sma_10, sma_20, ema_5, ema_10, rsi_14, atr_14])))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return percentage
    
    # Historical volatility calculation
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)  # Historical volatility (daily std)
    
    # Calculate thresholds based on volatility
    threshold = 2 * historical_vol  # Volatility-adaptive threshold for the reward system
    
    reward = 0
    
    # Reward adjustments based on recent return
    if recent_return < -threshold:
        reward -= 50  # High negative return
    elif recent_return > threshold:
        reward += 50  # High positive return
        
    # Assess other conditions (e.g., RSI, SMA, EMA)
    rsi = enhanced_s[39]  # Assuming RSI is at index 39 in the enhanced state
    if rsi < 30:
        reward += 10  # Oversold condition
    elif rsi > 70:
        reward -= 10  # Overbought condition
    
    # Example of additional risk consideration (not exceeding 5% risk)
    if recent_return < -5:
        reward -= 20  # If the recent return is very negative
    
    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]