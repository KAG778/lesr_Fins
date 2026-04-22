import numpy as np

def calculate_returns(prices):
    """ Calculate daily returns. """
    return np.diff(prices) / prices[:-1] * 100

def calculate_sma(prices, window):
    """ Calculate Simple Moving Average. """
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    """ Calculate Exponential Moving Average. """
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = [np.nan] * (window - 1) + [np.mean(prices[:window])]
    for price in prices[window:]:
        ema.append((price * alpha) + (ema[-1] * (1 - alpha)))
    return ema[-1]

def calculate_rsi(prices, window=14):
    """ Calculate Relative Strength Index. """
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
    """ Calculate MACD and Signal Line. """
    if len(prices) < 26:
        return np.nan, np.nan
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    signal_line = calculate_ema(prices[-9:], 9)  # Signal line is 9-day EMA of MACD
    return macd, signal_line

def calculate_atr(highs, lows, closes, window=14):
    """ Calculate Average True Range (ATR). """
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
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices)
    macd, signal_line = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices)

    # Calculate historical volatility
    returns = calculate_returns(closing_prices)
    historical_volatility = np.std(returns) if len(returns) > 0 else 0

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        [sma_5, sma_10, ema_5, rsi, macd, signal_line, historical_volatility, atr]
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_volatility = np.std(returns) if len(returns) > 0 else 1  # Avoid division by zero

    # Define reward initialization
    reward = 0
    
    # Use volatility-adaptive thresholds
    threshold = 2 * historical_volatility  # 2x historical volatility threshold
    
    # Adjust reward based on recent return
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return
    else:
        reward += 10  # Neutral or positive return

    # Trend indication using MACD
    macd = enhanced_s[-6]  # MACD value
    signal_line = enhanced_s[-5]  # Signal line value
    if macd > signal_line:
        reward += 20  # Indicates a potential uptrend
    elif macd < signal_line:
        reward -= 20  # Indicates a potential downtrend
    
    # Risk management using ATR
    atr = enhanced_s[-1]  # ATR value
    if np.abs(recent_return) > atr:
        reward -= 30  # Penalize high risk
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]