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

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    signal_line = calculate_ema(prices, 9)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    return np.std(np.diff(prices[-window:])) * 100  # Convert to percentage

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                   np.maximum(np.abs(highs[-window:] - closes[-window:-1]), 
                              np.abs(lows[-window:] - closes[-window:-1]))
    return np.mean(tr)

def revise_state(s):
    # Extracting raw OHLCV data
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]
    
    # Feature extraction
    features = []

    # Trend Indicators
    for window in [5, 10, 20, 50]:
        features.append(calculate_sma(closing_prices, window))
        features.append(calculate_ema(closing_prices, window))
    
    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    macd, signal, histogram = calculate_macd(closing_prices)
    features.extend([rsi_5, rsi_10, rsi_14, macd, signal, histogram])
    
    # Volatility Indicators
    historical_volatility_5 = calculate_volatility(closing_prices, 5)
    historical_volatility_20 = calculate_volatility(closing_prices, 20)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)  # 14-day ATR
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else np.nan
    features.extend([historical_volatility_5, historical_volatility_20, atr, volatility_ratio])
    
    # Volume Indicators
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], 
                             np.where(np.diff(closing_prices) < 0, -volumes[1:], 0)))
    volume_avg_5 = calculate_sma(volumes, 5)
    volume_avg_20 = calculate_sma(volumes, 20)
    volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) != 0 else np.nan
    features.extend([obv[-1], volume_avg_5, volume_avg_20, volume_ratio])
    
    # Market Regime Detection
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]  # Linear regression slope
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = (volume_avg_5 / volume_avg_20) > 2.0 if volume_avg_20 != 0 else np.nan
    
    features.extend([trend_strength, price_position, volume_ratio_regime])

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Position flag (1.0 = holding, 0.0 = not holding)
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Percentage return

    # Determine historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_vol = np.std(returns) * 100  # Convert to percentage
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if enhanced_s[5] < 30 and recent_return > threshold:  # RSI < 30 indicates oversold
            reward += 50  # Strong buy signal
        elif recent_return < -threshold:
            reward -= 20  # Penalize for potential missed opportunity
    elif position_flag == 1:  # Holding
        if recent_return < -threshold:  # If recent return is significantly negative
            reward -= 50  # Penalize for holding during a downturn
        elif enhanced_s[5] > 70 and recent_return > 0:  # RSI > 70 indicates overbought
            reward += 20  # Positive reward for holding

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]