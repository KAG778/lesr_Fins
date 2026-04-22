import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    return np.dot(weights, prices[-window:])

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices[-window:])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum.reduce([highs[-window:] - lows[-window:], 
                            np.abs(highs[-window:] - closes[-window:]), 
                            np.abs(lows[-window:] - closes[-window:])])
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    
    enhanced_s = np.zeros(120 + 6)  # Original 120 + 6 new features
    
    # Copy original state
    enhanced_s[0:120] = s

    # Calculate new features
    enhanced_s[120] = calculate_sma(closing_prices, 5)  # 5-day SMA
    enhanced_s[121] = calculate_sma(closing_prices, 10)  # 10-day SMA
    enhanced_s[122] = calculate_ema(closing_prices, 5)  # 5-day EMA
    enhanced_s[123] = calculate_rsi(closing_prices, 14)  # 14-day RSI
    enhanced_s[124] = calculate_atr(high_prices, low_prices, closing_prices, 14)  # 14-day ATR
    enhanced_s[125] = np.std(np.diff(closing_prices) / closing_prices[:-1])  # Volatility

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = enhanced_s[125]  # Volatility from revised state
    threshold = 2 * historical_volatility  # 2x historical volatility as threshold

    reward = 0
    
    # Determine reward based on recent return and volatility
    if recent_return > threshold:
        reward += 50  # Strong positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong negative momentum

    # Adjust reward based on RSI
    rsi = enhanced_s[123]
    if rsi < 30:  # Oversold condition
        reward += 25
    elif rsi > 70:  # Overbought condition
        reward -= 25
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]