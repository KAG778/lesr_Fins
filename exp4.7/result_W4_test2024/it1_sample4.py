import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window+1:]:
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

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    enhanced_s = np.copy(s)
    
    # Calculate indicators
    enhanced_s = np.concatenate((enhanced_s,
                                  [calculate_sma(closing_prices, 5)],  # 5-day SMA
                                  [calculate_ema(closing_prices, 10)],  # 10-day EMA
                                  [calculate_rsi(closing_prices, 14)],  # 14-day RSI
                                  [calculate_atr(high_prices, low_prices, closing_prices, 14)],  # ATR
                                  [np.std(np.diff(closing_prices) / closing_prices[:-1])]))  # Historical Volatility

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = enhanced_s[-1]  # Last feature is historical volatility
    threshold = 2 * historical_vol  # Use 2x historical volatility for adaptive threshold

    reward = 0

    # Reward logic based on recent return
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement
    else:
        reward += 10  # Mild return, hold position

    # Trend check using SMA and EMA
    sma_5 = enhanced_s[-5]  # Assuming last added feature is SMA 5
    ema_10 = enhanced_s[-6]  # Assuming second last added feature is EMA 10
    if sma_5 > ema_10:
        reward += 20  # Positive trend
    elif sma_5 < ema_10:
        reward -= 20  # Negative trend

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range