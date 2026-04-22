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

def calculate_macd(prices, fast_window=12, slow_window=26, signal_window=9):
    if len(prices) < slow_window:
        return np.nan, np.nan
    ema_fast = calculate_ema(prices, fast_window)
    ema_slow = calculate_ema(prices, slow_window)
    macd = ema_fast - ema_slow
    signal = calculate_ema(prices[-signal_window:], signal_window)
    return macd, signal

def calculate_atr(highs, lows, closes, window=14):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-1] - lows[-1], 
                    np.maximum(abs(highs[-1] - closes[-2]), abs(lows[-1] - closes[-2])))
    atr = np.mean(tr)
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:119]

    enhanced_s = np.concatenate((s, np.zeros(20)))  # Create space for new features

    # Calculate technical indicators
    enhanced_s[120] = calculate_sma(closing_prices, 5)
    enhanced_s[121] = calculate_sma(closing_prices, 10)
    enhanced_s[122] = calculate_sma(closing_prices, 20)
    enhanced_s[123] = calculate_ema(closing_prices, 5)
    enhanced_s[124] = calculate_ema(closing_prices, 10)
    enhanced_s[125] = calculate_rsi(closing_prices, 14)
    enhanced_s[126], enhanced_s[127] = calculate_macd(closing_prices)
    enhanced_s[128] = calculate_atr(high_prices, low_prices, closing_prices)

    # Handle edge cases
    enhanced_s = np.nan_to_num(enhanced_s)  # Replace NaNs with zeros for simplicity

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(historical_returns) if len(historical_returns) > 0 else 0

    threshold = 2 * historical_vol  # Volatility-adaptive threshold
    reward = 0

    # Reward conditions
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement
    else:
        reward += 10  # Mild return, hold position

    # Risk assessment based on volatility
    if historical_vol > 3:  # High volatility threshold
        reward -= 20  # Penalize riskier conditions

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range