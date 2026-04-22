import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

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
    rs = gain / loss if loss > 0 else 0
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
    high_prices = s[40:60]
    low_prices = s[60:80]

    # Calculate key features
    enhanced_s = np.copy(s)

    # Add trend features
    enhanced_s = np.concatenate((enhanced_s, [
        calculate_sma(closing_prices, 5),    # 5-day SMA
        calculate_sma(closing_prices, 10),   # 10-day SMA
        calculate_ema(closing_prices, 5),     # 5-day EMA
        calculate_rsi(closing_prices, 14),    # 14-day RSI
        calculate_atr(high_prices, low_prices, closing_prices, 14) # 14-day ATR
    ]))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    rsi = enhanced_s[100]  # Last RSI value
    atr = enhanced_s[104]  # Last ATR value
    
    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 1e-10  # Avoid division by zero

    # Use relative thresholds based on historical volatility
    threshold = 2 * historical_vol

    reward = 0

    # Reward based on recent return and RSI
    if recent_return > threshold:
        reward += 50  # High return
    elif recent_return < -threshold:
        reward -= 50  # Significant loss

    # Penalize if RSI indicates overbought or oversold
    if rsi > 70:
        reward -= 30  # Overbought
    elif rsi < 30:
        reward += 20  # Oversold (potential buy signal)

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]