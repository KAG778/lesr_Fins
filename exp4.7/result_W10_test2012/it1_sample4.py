import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])  # Initialize EMA with SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window - 1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Handle edge cases with NaN padding
    padded_sma_5 = np.pad(sma_5, (4, 0), constant_values=np.nan)
    padded_sma_10 = np.pad(sma_10, (9, 0), constant_values=np.nan)
    padded_ema_5 = np.pad(ema_5, (4, 0), constant_values=np.nan)
    padded_rsi = np.pad(np.array([rsi_14]), (13, 0), constant_values=np.nan)
    padded_atr = np.pad(atr, (13, 0), constant_values=np.nan)
    
    # Combine into enhanced state
    enhanced_s = np.concatenate([s, 
                                  padded_sma_5, 
                                  padded_sma_10, 
                                  padded_ema_5, 
                                  padded_rsi, 
                                  padded_atr])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 0 else 1  # Avoid division by zero
    threshold = 2 * historical_vol  # Adaptive threshold based on volatility

    reward = 0

    # Reward structure based on recent return and trend indicators
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum

    # Additional checks using RSI for risk assessment
    rsi = enhanced_s[39]  # Assuming the RSI is at this position
    if rsi < 30:
        reward += 10  # Oversold condition
    elif rsi > 70:
        reward -= 10  # Overbought condition

    # Additional trend check using SMA
    sma_5 = np.mean(closing_prices[-5:])
    sma_10 = np.mean(closing_prices[-10:])
    if sma_5 > sma_10:
        reward += 20  # Indicate an uptrend
    else:
        reward -= 20  # Indicate a downtrend

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]