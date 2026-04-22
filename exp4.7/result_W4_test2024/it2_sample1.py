import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]  # Start with the last price in the window
    for price in prices[-window + 1:]:
        ema = (price * alpha) + (ema * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window=14):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-1] - lows[-1], 
                    np.maximum(abs(highs[-1] - closes[-2]), abs(lows[-1] - closes[-2])))
    return np.mean(tr)

def calculate_volatility(prices):
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) if len(returns) > 0 else 0

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Create enhanced state with existing features
    enhanced_s = np.copy(s)

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr = calculate_atr(high_prices, lows, closing_prices, 14)
    historical_volatility = calculate_volatility(closing_prices)

    # Append new features to enhanced state
    enhanced_s = np.concatenate((enhanced_s, [sma_5, sma_10, ema_10, rsi_14, atr, historical_volatility]))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = enhanced_s[-1]  # Last feature is the calculated historical volatility

    # Define adaptive thresholds based on historical volatility
    threshold_up = 2 * historical_volatility  # 2x historical volatility for positive
    threshold_down = -2 * historical_volatility  # -2x historical volatility for negative

    reward = 0

    # Reward based on recent return
    if recent_return > threshold_up:
        reward += 50  # Strong upward movement
    elif recent_return < threshold_down:
        reward -= 50  # Strong downward movement

    # Trend evaluation using moving averages (sma_5 and sma_10)
    sma_5 = enhanced_s[-5]  # 5-day SMA
    sma_10 = enhanced_s[-4]  # 10-day SMA
    if sma_5 > sma_10:
        reward += 20  # Positive trend (bullish)
    elif sma_5 < sma_10:
        reward -= 20  # Negative trend (bearish)

    return np.clip(reward, -100, 100)  # Ensure reward stays within the specified range