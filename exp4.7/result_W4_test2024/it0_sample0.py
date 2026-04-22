import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window):
    sma = calculate_sma(prices, window)
    rolling_std = np.std(prices[-window:])
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    return upper_band, lower_band

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-window:])
    return atr

def revise_state(s):
    enhanced_s = np.copy(s)
    
    # Closing prices
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate indicators
    # 5-day SMA
    sma_5 = calculate_sma(closing_prices, 5)
    enhanced_s = np.concatenate((enhanced_s, sma_5[-1:] if len(sma_5) > 0 else np.array([np.nan])))

    # 10-day SMA
    sma_10 = calculate_sma(closing_prices, 10)
    enhanced_s = np.concatenate((enhanced_s, sma_10[-1:] if len(sma_10) > 0 else np.array([np.nan])))

    # 14-day RSI
    rsi = calculate_rsi(closing_prices, 14)
    enhanced_s = np.concatenate((enhanced_s, np.array([rsi])))

    # Bollinger Bands
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20)
    enhanced_s = np.concatenate((enhanced_s, np.array([upper_band[-1]]), np.array([lower_band[-1]])))

    # Average True Range (ATR)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    enhanced_s = np.concatenate((enhanced_s, np.array([atr])))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    
    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol
    
    reward = 0
    
    # Reward logic based on recent return and trends
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement
    elif np.all(closing_prices[-5:] > closing_prices[-10:-5]):
        reward += 20  # Uptrend
    elif np.all(closing_prices[-5:] < closing_prices[-10:-5]):
        reward -= 20  # Downtrend
    else:
        reward -= 10  # Sideways market

    # Risk control based on ATR
    atr = enhanced_s[-1]
    if recent_return < -1.5 * atr:  # 1.5x ATR threshold for significant loss
        reward -= 30  # High risk position
    
    return np.clip(reward, -100, 100)