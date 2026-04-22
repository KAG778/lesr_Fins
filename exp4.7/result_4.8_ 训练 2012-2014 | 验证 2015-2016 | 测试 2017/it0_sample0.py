import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema[-1]

def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs, lows, closes, period=14):
    if len(highs) < period:
        return np.nan
    tr = np.maximum(highs[-period:] - lows[-period:], 
                    np.abs(highs[-period:] - closes[-period:]), 
                    np.abs(lows[-period:] - closes[-period:]))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:119]

    enhanced_s = np.copy(s)

    # Calculate additional technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi = calculate_rsi(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices)

    # Add new features to enhanced state
    enhanced_s = np.concatenate((enhanced_s, [sma_5, sma_10, sma_20, ema_5, ema_10, rsi, atr]))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Daily returns in %

    # Calculate thresholds based on historical volatility
    threshold = 2 * historical_vol

    # Initialize reward
    reward = 0

    # Assess trading conditions
    if recent_return > threshold:
        reward += 50  # Positive reward for good upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative reward for significant downward movement

    # Example of controlling risk
    if np.abs(recent_return) > 5:  # 5% as a safe risk threshold
        reward -= 30

    # Incorporate RSI for risk assessment
    rsi = enhanced_s[-2]  # Assuming RSI is the second last new feature
    if rsi > 70:
        reward -= 20  # Overbought condition
    elif rsi < 30:
        reward += 20  # Oversold condition

    return np.clip(reward, -100, 100)  # Ensure the reward is within the specified range