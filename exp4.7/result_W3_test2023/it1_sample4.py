import numpy as np

def compute_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def compute_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def compute_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def compute_macd(prices):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_12 = compute_ema(prices, 12)
    ema_26 = compute_ema(prices, 26)
    return ema_12[-len(ema_26):] - ema_26

def compute_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.convolve(tr, np.ones(window) / window, mode='valid')
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Calculate technical indicators
    sma_5 = compute_sma(closing_prices, 5)
    sma_10 = compute_sma(closing_prices, 10)
    rsi_14 = compute_rsi(closing_prices, 14)
    macd = compute_macd(closing_prices)
    atr_14 = compute_atr(high_prices, lows, closing_prices, 14)

    # Create enhanced state with more robust features
    enhanced_s = np.concatenate([
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        volumes,
        adjusted_closing_prices,
        sma_5[-1:], sma_10[-1:],
        [rsi_14],
        [macd[-1]],
        [atr_14[-1]]
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility

    # Set adaptive thresholds based on historical volatility
    threshold_positive = 2 * historical_vol
    threshold_negative = -2 * historical_vol

    # Reward calculation
    reward = 0

    # Reward structure based on returns and volatility
    if recent_return > threshold_positive:
        reward += 50  # Strong positive return
    elif recent_return < threshold_negative:
        reward -= 50  # Strong negative return

    # Add conditions based on RSI
    rsi = enhanced_s[-4]  # Assuming RSI is the 5th last feature in the enhanced state
    if rsi < 30:
        reward += 20  # Potential buy opportunity
    elif rsi > 70:
        reward -= 20  # Potential sell opportunity

    # Risk management based on recent return
    if recent_return < threshold_negative and historical_vol > 0:
        reward -= 30  # Penalty for excessive loss

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]