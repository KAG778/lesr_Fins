import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Start with the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros_like(closes)
    atr[window-1] = np.mean(tr[:window])  # The first ATR value
    for i in range(window, len(closes)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)

    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Combine all features into a new state
    enhanced_s = np.concatenate((
        closing_prices, opening_prices, high_prices, low_prices,
        volumes, adjusted_closing_prices,
        sma_5[-1:], sma_10[-1:], sma_20[-1:],  # Only take the last value for each SMA
        ema_5[-1:], ema_10[-1:],  # Only the last EMA
        [rsi_14],  # Last RSI
        [atr_14[-1]]  # Last ATR
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)

    # Set thresholds based on historical volatility
    threshold = 2 * historical_volatility  # Relative threshold

    reward = 0
    
    # Reward for upward momentum
    if recent_return > threshold:
        reward += 50  # Good upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Bad downward momentum

    # Incorporate RSI for overbought/oversold conditions
    rsi_value = enhanced_s[100]  # Assuming RSI is at index 100
    if rsi_value > 70:
        reward -= 20  # Overbought condition
    elif rsi_value < 30:
        reward += 20  # Oversold condition

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]