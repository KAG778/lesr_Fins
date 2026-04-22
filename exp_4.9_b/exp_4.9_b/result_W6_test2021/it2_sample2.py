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
    ema[window-1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema[-1]

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    if np.isnan(ema_12) or np.isnan(ema_26):
        return np.nan, np.nan, np.nan
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(prices[-26:], 9)  # Signal line is EMA of MACD line
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.maximum(np.abs(highs[-window:] - closes[-window:]), 
                               np.abs(lows[-window:] - closes[-window:])))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    features = []

    # Trend Indicators
    features.append(calculate_sma(closing_prices, 5))      # 5-day SMA
    features.append(calculate_sma(closing_prices, 10))     # 10-day SMA
    features.append(calculate_sma(closing_prices, 20))     # 20-day SMA
    features.append(calculate_ema(closing_prices, 5))      # 5-day EMA
    features.append(calculate_ema(closing_prices, 10))     # 10-day EMA
    features.append(calculate_ema(closing_prices, 20))     # 20-day EMA

    # Momentum Indicators
    features.append(calculate_rsi(closing_prices, 5))      # 5-day RSI
    features.append(calculate_rsi(closing_prices, 14))     # 14-day RSI
    macd_line, signal_line, histogram = calculate_macd(closing_prices)
    features.append(macd_line)                               # MACD line
    features.append(signal_line)                             # MACD signal line
    features.append(histogram)                               # MACD histogram

    # Volatility Indicators
    features.append(np.std(np.diff(closing_prices[-5:])) * np.sqrt(252))  # 5-day historical volatility
    features.append(calculate_atr(high_prices, low_prices, closing_prices, 14))  # ATR
    features.append(np.std(np.diff(closing_prices[-20:])) / np.std(np.diff(closing_prices)) if np.std(np.diff(closing_prices)) != 0 else np.nan)  # Volatility ratio

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    features.append(obv[-1])  # Most recent OBV
    features.append(volumes[-1] / np.mean(volumes[-20:]))  # Volume ratio
    features.append(np.corrcoef(volumes[-20:], closing_prices[-20:])[0, 1])  # Volume-price correlation

    # Market Regime Detection
    volatility_ratio = np.std(np.diff(closing_prices[-5:])) / np.std(np.diff(closing_prices[-20:])) if np.std(np.diff(closing_prices[-20:])) != 0 else np.nan
    features.append(volatility_ratio)                         # Volatility ratio
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]**2  # R^2 for trend
    features.append(trend_strength)                           # Trend strength
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    features.append(price_position)                           # Price position
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan
    features.append(volume_ratio_regime)                     # Volume ratio regime

    # Generating new features
    features.append(np.mean(closing_prices[-5:]))            # 5-day average closing price
    features.append(np.max(closing_prices[-5:]))             # 5-day max closing price
    features.append(np.min(closing_prices[-5:]))             # 5-day min closing price
    features.append(np.mean(np.diff(closing_prices[-5:])) * 100)  # 5-day average daily return
    features.append(np.std(np.diff(closing_prices[-5:])) * 100)   # 5-day volatility
    features.append(np.mean(opening_prices[-5:]))            # 5-day average opening price
    features.append(np.mean(high_prices[-5:]))               # 5-day average high price
    features.append(np.mean(low_prices[-5:]))                # 5-day average low price
    features.append(np.mean(volumes[-5:]))                   # 5-day average volume
    features.append(np.mean(closing_prices[-5:]) / np.mean(closing_prices[-20:]))  # 5-day average vs 20-day average

    # Combine with original state
    enhanced_s = np.concatenate((s, np.array(features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100  # Recent returns in percentage

    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Volatility adaptive threshold

    recent_return = returns[-1] if len(returns) > 0 else 0  # Latest return in percentage
    reward = 0

    # Reward structure based on position
    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50
    elif position_flag == 1:  # Holding
        if recent_return > 0:  # Positive return
            reward += 20
        elif recent_return < -threshold:  # Significant drop
            reward -= 50

    # Penalize uncertain market conditions
    if np.isnan(recent_return) or np.std(returns) < (0.5 * historical_vol):
        reward -= 20  # Choppy market

    return np.clip(reward, -100, 100)  # Ensure reward is within limits