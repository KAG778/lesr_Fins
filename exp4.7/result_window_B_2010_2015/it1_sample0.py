import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window + 1:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices[-window:])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss > 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                   np.abs(highs[-window:] - closes[-window:]), 
                   np.abs(lows[-window:] - closes[-window:]))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    enhanced_s = np.zeros(130)  # Original 120 + 10 new features

    # Copy original state
    enhanced_s[:120] = s

    # Calculate new features
    enhanced_s[120] = calculate_sma(closing_prices, 5)   # 5-day SMA
    enhanced_s[121] = calculate_sma(closing_prices, 10)  # 10-day SMA
    enhanced_s[122] = calculate_ema(closing_prices, 5)   # 5-day EMA
    enhanced_s[123] = calculate_ema(closing_prices, 10)  # 10-day EMA
    enhanced_s[124] = calculate_rsi(closing_prices, 14)   # 14-day RSI
    enhanced_s[125] = calculate_atr(high_prices, low_prices, closing_prices, 14)  # 14-day ATR
    enhanced_s[126] = np.mean(volumes[-5:])  # 5-day average volume
    enhanced_s[127] = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Recent return
    enhanced_s[128] = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    enhanced_s[129] = np.mean(volumes[-10:])  # 10-day average volume

    return enhanced_s

def intrinsic_reward(enhanced_s):
    recent_return = enhanced_s[127]  # Recent return
    historical_volatility = enhanced_s[128]  # Historical volatility
    
    # Avoid division by zero
    if historical_volatility == 0:
        return 0

    threshold = 2 * historical_volatility  # Relative threshold based on historical volatility
    reward = 0

    # Reward for positive recent returns
    if recent_return > threshold:
        reward += 50  # Strong upward momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong downward momentum

    # Additional penalty for high volatility
    if historical_volatility > 5:  # Assuming 5% is a high-risk threshold
        reward -= 20  # Penalize for high volatility

    return np.clip(reward, -100, 100)  # Clamp reward to [-100, 100]