import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    short_ema = calculate_ema(prices, 12)[-1]  # 12-day EMA
    long_ema = calculate_ema(prices, 26)[-1]   # 26-day EMA
    return short_ema - long_ema

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        sma_5[-1:],  # last value of 5-day SMA
        sma_10[-1:],  # last value of 10-day SMA
        [rsi],
        [macd],
        [atr]
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return
    historical_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(historical_returns)

    # Calculate threshold
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    # Assess reward based on recent return relative to historical volatility
    if recent_return < -threshold:
        reward -= 50  # High loss relative to volatility
    elif recent_return > threshold:
        reward += 50  # Strong gain relative to volatility

    # Additional conditions can be added here for further granularity in reward calculation
    # For instance, checking trend direction based on SMA or momentum indicators

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range