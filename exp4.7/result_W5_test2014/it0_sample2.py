import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    ema = [np.nan] * (window - 1) + [np.mean(prices[:window])]
    for price in prices[window:]:
        ema.append((price * (2/(window + 1))) + (ema[-1] * (1 - (2/(window + 1)))))
    return ema[-1]

def calculate_rsi(prices, window=14):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    if avg_loss == 0:
        return 100  # Avoid division by zero
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    if len(prices) < 26:
        return np.nan, np.nan  # Need at least 26 prices for MACD
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    signal_line = calculate_ema(prices[-9:], 9)  # Signal line is 9-day EMA of MACD
    return macd, signal_line

def calculate_atr(highs, lows, closes, window=14):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.maximum(np.abs(highs[-window:] - closes[-window:]), 
                               np.abs(lows[-window:] - closes[-window:])))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices)
    macd, signal_line = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices)
    
    # Create enhanced state
    enhanced_s = np.concatenate((s, 
                                  [sma_5, sma_10, sma_20, ema_5, rsi, macd, signal_line, atr]))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    
    # Set reward thresholds
    threshold = 2 * historical_vol
    
    reward = 0
    
    if recent_return > threshold:
        reward += 50  # Positive trend
    elif recent_return < -threshold:
        reward -= 50  # Negative trend
    else:
        reward -= 20  # Sideways movement
    
    # Risk management
    if np.abs(recent_return) > 5:  # 5% is a hard limit for daily loss
        reward -= 30  # Penalize high risk
    
    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]