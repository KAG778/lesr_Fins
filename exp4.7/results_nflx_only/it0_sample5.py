import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[window-1] = np.mean(prices[:window])  # First EMA is an SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if window < len(gain) else np.mean(gain)
    avg_loss = np.mean(loss[-window:]) if window < len(loss) else np.mean(loss)
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(abs(highs[1:] - closes[:-1]), 
                               abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i-1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volumes = s[80:99]

    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi_14 = np.array([calculate_rsi(closing_prices, 14) if i >= 14 else np.nan for i in range(len(closing_prices))])
    atr_14 = np.array([calculate_atr(high_prices, low_prices, closing_prices, 14) if i >= 14 else np.nan for i in range(len(closing_prices))])

    # Create enhanced state (original + new features)
    enhanced_s = np.concatenate((
        s,
        sma_5[-20:],  # 20 most recent values
        sma_10[-20:],
        ema_5[-20:],
        rsi_14[-20:],
        atr_14[-20:],
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    
    # Reward calculation
    reward = 0
    threshold = 2 * historical_volatility  # Adaptive threshold

    if recent_return > threshold:
        reward += 50  # Strong positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong negative momentum

    # Add additional conditions for sideways movement, controlled risk, etc.
    # Example: if RSI is between 30 and 70, we consider the market to be stable
    rsi = enhanced_s[100:120]  # Assuming RSI is the last 20 dimensions from enhanced state
    if np.all((rsi >= 30) & (rsi <= 70)):
        reward += 20  # Favorable trading conditions

    return np.clip(reward, -100, 100)  # Ensure reward is within range [-100, 100]