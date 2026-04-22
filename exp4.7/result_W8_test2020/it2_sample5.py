import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Start with the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else np.mean(gain)
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else np.mean(loss)

    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])  # Average over the last 'window'

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    trading_volume = s[80:100]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1]  # Latest 5-day SMA
    sma_10 = calculate_sma(closing_prices, 10)[-1]  # Latest 10-day SMA
    ema_5 = calculate_ema(closing_prices, 5)[-1]  # Latest 5-day EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # Latest RSI
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)  # Latest ATR

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        [sma_5, sma_10, ema_5, rsi_14, atr_14]
    ))

    # Fill any missing values with NaN
    while len(enhanced_s) < 120:
        enhanced_s = np.append(enhanced_s, [np.nan])
    
    return enhanced_s[:120]

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_volatility = np.std(returns) if len(returns) > 0 else 0

    # Use volatility-adaptive thresholds
    threshold = 2 * historical_volatility  # 2x historical volatility

    reward = 0

    # Reward based on recent return and volatility threshold
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Adjust reward based on RSI
    rsi = enhanced_s[-1]  # Assuming the last index is RSI
    if rsi < 30:
        reward += 20  # Oversold condition, potential buy signal
    elif rsi > 70:
        reward -= 20  # Overbought condition, potential sell signal

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]