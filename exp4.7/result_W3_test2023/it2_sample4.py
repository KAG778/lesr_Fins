import numpy as np

def compute_moving_average(prices, window):
    """Calculate Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def compute_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(highs, lows, closes, window=14):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = compute_moving_average(tr, window)
    return np.concatenate((np.full(window-1, np.nan), atr))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate technical indicators
    sma_5 = compute_moving_average(closing_prices, 5)
    sma_10 = compute_moving_average(closing_prices, 10)
    rsi_14 = np.concatenate((np.full(13, np.nan), [compute_rsi(closing_prices, 14)]))  # Handle edge case for RSI
    atr_14 = compute_atr(high_prices, low_prices, closing_prices, 14)

    # Create enhanced state with selected features
    enhanced_s = np.concatenate([
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        volumes,
        sma_5[-1:], 
        sma_10[-1:], 
        [rsi_14[-1]], 
        [atr_14[-1]]
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility

    # Use adaptive thresholds
    threshold = 2 * historical_vol  # 2x historical volatility as threshold

    reward = 0
    
    # Reward based on return and volatility
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Adjustments based on RSI
    rsi = enhanced_s[-2]  # Assuming RSI is the second last feature
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    # Risk control: penalty for excessive loss
    if recent_return < -threshold and historical_vol > 0:  # Volatility must be positive
        reward -= 30  # Penalty for excessive loss based on volatility

    return np.clip(reward, -100, 100)  # Ensure reward stays within bounds