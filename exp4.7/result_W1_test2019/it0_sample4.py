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
    for price in prices[-window+1:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    if len(prices) < 26:
        return np.nan
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    return ema_12 - ema_26

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-1] - lows[-1], 
                    np.abs(highs[-1] - closes[-2]), 
                    np.abs(lows[-1] - closes[-2]))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    enhanced_s = np.zeros(120 + 6)  # Original + 6 new features
    
    # Copy original state
    enhanced_s[0:120] = s
    
    # Calculate additional features
    enhanced_s[120] = calculate_sma(closing_prices, 5)  # 5-day SMA
    enhanced_s[121] = calculate_sma(closing_prices, 10)  # 10-day SMA
    enhanced_s[122] = calculate_sma(closing_prices, 20)  # 20-day SMA
    
    enhanced_s[123] = calculate_ema(closing_prices, 12)  # 12-day EMA
    enhanced_s[124] = calculate_rsi(closing_prices, 14)   # 14-day RSI
    enhanced_s[125] = calculate_macd(closing_prices)       # MACD

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # % return
    rsi = enhanced_s[124]  # 14-day RSI
    
    # Calculate historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(daily_returns) if len(daily_returns) > 0 else 0
    
    # Define a relative threshold for rewards
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    # Adjust reward based on recent return relative to volatility
    if recent_return < -threshold:
        reward -= 50
    elif recent_return > threshold:
        reward += 50

    # Adjust reward based on RSI
    if rsi < 30:  # Oversold
        reward += 20
    elif rsi > 70:  # Overbought
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]