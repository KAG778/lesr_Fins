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

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices, 14)
    
    # Calculate volatility
    returns = calculate_returns(closing_prices)
    historical_volatility = np.std(returns) if len(returns) > 0 else 0
    
    # Calculate Average True Range (ATR) for risk management
    atr = np.mean(np.maximum(high_prices[-14:] - low_prices[-14:], 
                             np.maximum(np.abs(high_prices[-14:] - closing_prices[-14:]), 
                                        np.abs(low_prices[-14:] - closing_prices[-14:])))
                   ) if len(high_prices) >= 14 else np.nan

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        [sma_5, sma_10, ema_5, rsi, historical_volatility, atr]
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = calculate_returns(closing_prices)
    volatility = np.std(returns) if len(returns) > 0 else 0
    
    # Define reward initialization
    reward = 0
    
    # Use volatility-adaptive thresholds
    threshold = 2 * volatility  # 2x historical volatility threshold
    
    # Adjust reward based on recent return
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return
    
    # Incorporate RSI into the reward structure
    rsi = enhanced_s[-5]  # Last RSI value
    if rsi < 30:
        reward += 20  # Oversold condition, potential buy signal
    elif rsi > 70:
        reward -= 20  # Overbought condition, potential sell signal
    
    # Risk management based on ATR
    atr = enhanced_s[-1]  # Last ATR value
    if np.abs(recent_return) > atr:
        reward -= 30  # Penalize high risk
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]