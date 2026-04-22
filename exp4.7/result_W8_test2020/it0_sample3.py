import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window-1] = np.mean(prices[:window])  # Initialize with SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = calculate_sma(gain, window)
    avg_loss = calculate_sma(loss, window)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([np.full(window-1, np.nan), rsi])  # Prepend NaNs for alignment

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volumes = s[80:99]
    
    # Calculate additional features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices)

    # Combine features into enhanced state
    enhanced_s = np.concatenate([
        s,
        sma_5[-20:].reshape(-1),  # Last 20 values of SMA
        sma_10[-20:].reshape(-1),  # Last 20 values of SMA
        sma_20[-20:].reshape(-1),  # Last 20 values of SMA
        ema_5[-20:].reshape(-1),  # Last 20 values of EMA
        rsi[-20:].reshape(-1)  # Last 20 values of RSI
    ])

    # Handle NaNs with zeros or some other method if needed
    enhanced_s = np.nan_to_num(enhanced_s)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    rsi = enhanced_s[120:140]  # Assuming RSI is in the last 20 dimensions of enhanced_s

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_volatility = np.std(returns)

    # Set thresholds based on historical volatility
    threshold = 2 * historical_volatility  # Volatility-adaptive threshold

    reward = 0

    # Reward based on recent return
    if recent_return > threshold:  # Strong upward movement
        reward += 50
    elif recent_return < -threshold:  # Strong downward movement
        reward -= 50

    # Penalize if RSI indicates overbought or oversold
    if rsi[-1] > 70:  # Overbought
        reward -= 20
    elif rsi[-1] < 30:  # Oversold
        reward += 20

    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]