import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros(len(closes))
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(closes)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volumes = s[80:99]
    adjusted_closing_prices = s[100:119]
    
    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Handle edge cases for SMA and EMA by padding with NaNs
    sma_5 = np.concatenate((np.full(4, np.nan), sma_5))
    sma_10 = np.concatenate((np.full(9, np.nan), sma_10))
    ema_5 = np.concatenate((np.full(4, np.nan), ema_5))

    # Create enhanced state
    enhanced_s = np.concatenate((s, sma_5, sma_10, ema_5, np.array([rsi_14]), atr_14))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # recent return in percentage
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # historical volatility

    # Set reward parameters
    reward = 0
    threshold = 2 * historical_volatility if historical_volatility != 0 else 0
    
    # Reward for clear trends
    if (enhanced_s[20] > enhanced_s[21]) and (enhanced_s[21] > enhanced_s[22]):  # Example condition for uptrend
        reward += 30  # Reward for uptrend

    elif (enhanced_s[20] < enhanced_s[21]) and (enhanced_s[21] < enhanced_s[22]):  # Example condition for downtrend
        reward -= 30  # Penalty for downtrend

    # Adjust reward based on recent return
    if recent_return < -threshold:
        reward -= 50  # Strong penalty for large negative returns
    elif recent_return > threshold:
        reward += 50  # Strong reward for large positive returns

    return np.clip(reward, -100, 100)  # Ensure reward stays within bounds