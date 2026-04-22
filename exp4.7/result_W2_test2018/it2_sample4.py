import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]  # Start EMA with the first data point
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros(len(closes))
    atr[window - 1] = np.mean(tr[:window])  # Initial ATR value
    for i in range(window, len(atr)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i - 1]) / window  # SMA of TR
    return atr

def calculate_momentum(prices, window):
    """Calculate momentum as the difference between current and past price."""
    return prices[-1] - prices[-window] if len(prices) >= window else 0

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    rsi = calculate_rsi(closing_prices, 14)  # 14-day RSI
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)[-1]  # 14-day ATR
    momentum = calculate_momentum(closing_prices, 5)  # 5-day momentum

    # Combine into enhanced state
    enhanced_s = np.concatenate((s,
                                  [sma_5, sma_10, ema_5, rsi, atr, momentum]))

    # Handle edge cases
    enhanced_s = np.nan_to_num(enhanced_s, nan=0.0)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return

    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)  # Historical volatility

    # Define adaptive thresholds
    threshold = 2 * historical_vol  # Use 2x historical volatility as threshold
    
    # Initialize reward
    reward = 0
    
    # Reward logic based on recent return, RSI, and momentum
    rsi = enhanced_s[100]  # Last value of RSI
    momentum = enhanced_s[105]  # Last value of momentum

    if recent_return > threshold and momentum > 0:  # Favorable buying condition
        reward += 50
    elif recent_return < -threshold and momentum < 0:  # Unfavorable selling condition
        reward -= 50
    elif 30 < rsi < 70:  # Neutral but stable condition
        reward += 10  # Mild positive reward

    # Additional reward logic based on RSI levels
    if rsi < 30:  # Oversold condition
        reward += 20
    elif rsi > 70:  # Overbought condition
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]