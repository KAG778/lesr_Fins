import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]  # Start with the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = calculate_sma(gain, window)
    avg_loss = calculate_sma(loss, window)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate((np.full(window - 1, np.nan), rsi))  # Prepend NaN for alignment

def calculate_macd(prices):
    """Calculate MACD."""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12[11:] - ema_26[25:]  # Align lengths
    signal_line = calculate_ema(macd, 9)  # Signal line for MACD
    return macd, signal_line

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    macd, signal_line = calculate_macd(closing_prices)

    # Prepare enhanced state
    enhanced_s = np.concatenate((
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        volume,
        sma_5[-20:],  # Last 20 for alignment
        sma_10[-20:],
        sma_20[-20:],
        ema_5[-20:],
        ema_10[-20:],
        rsi_14[-20:],
        macd[-20:],  # MACD values
        signal_line[-20:]  # Signal line values
    ))

    # Ensure dimensions match
    if enhanced_s.shape[0] != 120 + 12:  # Original 120 + 12 new features
        raise ValueError("Enhanced state does not have the expected dimensions.")

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Recent return in %
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0.01  # Avoid division by zero, set a small default

    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol

    # Initialize reward
    reward = 0
    
    # Reward based on recent return relative to threshold
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return

    # Incorporate trend analysis using SMA
    sma_5 = enhanced_s[120:140]  # Last 20 elements for SMA 5
    sma_10 = enhanced_s[140:160]  # Last 20 elements for SMA 10
    if sma_5[-1] > sma_10[-1]:
        reward += 20  # Uptrend
    elif sma_5[-1] < sma_10[-1]:
        reward -= 20  # Downtrend

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]