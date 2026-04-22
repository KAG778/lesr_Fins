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

def calculate_rsi(prices, window=14):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices[-window:])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss > 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window=14):
    if len(highs) < window:
        return np.nan
    tr = np.maximum.reduce([highs[-window:] - lows[-window:], 
                            np.abs(highs[-window:] - closes[-window:]), 
                            np.abs(lows[-window:] - closes[-window:])])
    return np.mean(tr)

def calculate_macd(prices):
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    return ema12 - ema26

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    enhanced_s = np.zeros(126)  # Original state + new features

    # Copy original state
    enhanced_s[0:120] = s
    
    # Calculate new features
    enhanced_s[120] = calculate_sma(closing_prices, 5)  # 5-day SMA
    enhanced_s[121] = calculate_ema(closing_prices, 5)  # 5-day EMA
    enhanced_s[122] = calculate_rsi(closing_prices, 14)  # 14-day RSI
    enhanced_s[123] = calculate_atr(high_prices, low_prices, closing_prices, 14)  # 14-day ATR
    enhanced_s[124] = calculate_macd(closing_prices)  # MACD
    enhanced_s[125] = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0  # Volume change

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Recent return in percent
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1])  # Historical volatility
    threshold = 2 * historical_volatility  # Use 2x historical volatility as threshold

    reward = 0
    
    # Reward based on recent return relative to historical volatility
    if recent_return > threshold:
        reward += 50  # Strong positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong negative momentum

    # Adjust reward based on RSI
    rsi = enhanced_s[122]  # RSI from enhanced state
    if rsi < 30:  # Oversold condition
        reward += 25
    elif rsi > 70:  # Overbought condition
        reward -= 25

    # Add reward for volume momentum
    volume_momentum = enhanced_s[125]  # Volume change feature
    if volume_momentum > 0:  # Increased volume
        reward += 10
    elif volume_momentum < 0:  # Decreased volume
        reward -= 10
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]