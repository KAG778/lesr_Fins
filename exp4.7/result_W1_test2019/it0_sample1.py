import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
    rs = avg_gain / (avg_loss + 1e-10)  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    return ema_12[25:] - ema_26[25:]  # Adjust for initial EMA delays

def calculate_atr(prices, highs, lows, window):
    """Calculate Average True Range."""
    high_low = highs - lows
    high_prev_close = np.abs(highs[1:] - prices[:-1])
    low_prev_close = np.abs(lows[1:] - prices[:-1])
    tr = np.maximum(high_low[1:], np.maximum(high_prev_close, low_prev_close))
    atr = np.convolve(tr, np.ones(window)/window, mode='valid')
    return atr

def revise_state(s):
    """Enhance the state with additional features."""
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]
    
    # Create enhanced features
    sma_5 = calculate_sma(closing_prices, 5) if len(closing_prices) >= 5 else np.array([np.nan]*15)
    sma_10 = calculate_sma(closing_prices, 10) if len(closing_prices) >= 10 else np.array([np.nan]*10)
    sma_20 = calculate_sma(closing_prices, 20) if len(closing_prices) >= 20 else np.array([np.nan]*0)

    ema_5 = calculate_ema(closing_prices, 5)[4:] if len(closing_prices) >= 5 else np.array([np.nan]*15)
    rsi = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.array([np.nan]*6)
    macd = calculate_macd(closing_prices)

    # Calculate ATR
    atr = calculate_atr(closing_prices, high_prices, low_prices, 14) if len(closing_prices) >= 14 else np.array([np.nan]*6)
    
    # Combine all features (handling NaNs)
    enhanced_s = np.concatenate([
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        volume,
        np.pad(sma_5, (15, 0), mode='constant', constant_values=np.nan),
        np.pad(sma_10, (10, 0), mode='constant', constant_values=np.nan),
        np.pad(sma_20, (0, 0), mode='constant', constant_values=np.nan),
        np.pad(ema_5, (15, 0), mode='constant', constant_values=np.nan),
        np.pad(rsi, (6, 0), mode='constant', constant_values=np.nan),
        np.pad(macd, (25, 0), mode='constant', constant_values=np.nan),
        np.pad(atr, (6, 0), mode='constant', constant_values=np.nan),
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    """Calculate intrinsic reward based on enhanced state."""
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 1e-10  # Prevent division by zero

    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol

    # Calculate reward
    reward = 0

    # Reward for trending states
    if enhanced_s[95] > enhanced_s[90]:  # Assuming enhanced_s[95] is the latest SMA or EMA
        if recent_return > threshold:
            reward += 50  # Positive trend with good momentum
    elif enhanced_s[95] < enhanced_s[90]:
        if recent_return < -threshold:
            reward -= 50  # Negative trend with bad momentum

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range