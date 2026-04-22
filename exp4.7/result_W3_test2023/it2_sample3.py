import numpy as np

def compute_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def compute_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def compute_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.convolve(tr, np.ones(window)/window, mode='valid')
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate technical indicators
    sma_5 = compute_sma(closing_prices, 5)
    sma_10 = compute_sma(closing_prices, 10)
    rsi_14 = compute_rsi(closing_prices, 14)
    atr_14 = compute_atr(high_prices, lows, closing_prices, 14)
    
    # Enhance state representation
    enhanced_s = np.concatenate([
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        volumes,
        sma_5[-1:], 
        sma_10[-1:], 
        [rsi_14],
        atr_14[-1:]  # Include ATR for risk management
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Use 2x historical volatility as threshold
    
    # Initialize reward
    reward = 0
    
    # Reward based on recent return and volatility
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return

    # Further adjustments based on RSI
    rsi = enhanced_s[-2]  # Assuming RSI is the second last feature
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    # Incorporate risk management with the ATR
    atr = enhanced_s[-1]  # Assuming ATR is the last feature
    if recent_return < -1 * threshold and atr > 0:
        reward -= 30  # Penalty for excessive loss when volatility is high

    # Ensure reward stays within bounds
    reward = np.clip(reward, -100, 100)

    return reward