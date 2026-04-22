import numpy as np

def calculate_sma(prices, period):
    return np.convolve(prices, np.ones(period) / period, mode='valid')

def calculate_ema(prices, period):
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    alpha = 2 / (period + 1)
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-period:])

def revise_state(s):
    # Extract price and volume data from the raw state
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)  # 5-day SMA
    sma_10 = calculate_sma(closing_prices, 10)  # 10-day SMA
    sma_20 = calculate_sma(closing_prices, 20)  # 20-day SMA
    
    ema_5 = calculate_ema(closing_prices, 5)  # 5-day EMA
    ema_10 = calculate_ema(closing_prices, 10)  # 10-day EMA
    
    rsi = calculate_rsi(closing_prices, 14)  # 14-day RSI
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)  # 14-day ATR
    
    # Create enhanced state
    enhanced_s = np.concatenate((s, 
                                  sma_5[-1:], sma_10[-1:], sma_20[-1:], 
                                  ema_5[-1:], ema_10[-1:], 
                                  np.array([rsi]), 
                                  np.array([atr])))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # % return
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    threshold = 2 * historical_volatility  # Adaptive threshold

    reward = 0
    
    # Reward logic based on recent return and volatility
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return
    
    # Additional checks for risk management
    rsi = enhanced_s[-3]  # Last calculated RSI
    if rsi < 30:
        reward -= 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]