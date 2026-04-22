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
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema(prices, 9)  # Signal line is EMA of MACD
    histogram = macd_line - signal_line
    return (macd_line, signal_line, histogram)

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.maximum(np.abs(highs[-window:] - closes[-window:]), 
                               np.abs(lows[-window:] - closes[-window:])))
    return np.mean(tr)

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    enhanced_features = []

    # Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    macd_line, signal_line, histogram = calculate_macd(closing_prices)

    # Volatility Indicators
    atr_14 = calculate_atr(high_prices, lows, closing_prices, 14)
    historical_volatility_5 = calculate_volatility(closing_prices, 5)
    historical_volatility_20 = calculate_volatility(closing_prices, 20)
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else np.nan

    # Volume Indicators
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[:-1], -volumes[:-1]))
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Market Regime Detection
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2 if len(closing_prices) > 1 else np.nan
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # New Features
    enhanced_features.extend([
        sma_5, sma_10, sma_20, ema_5, ema_10,
        rsi_5, rsi_10,
        macd_line, signal_line, histogram,
        atr_14, historical_volatility_5, historical_volatility_20, volatility_ratio,
        obv[-1], volume_ratio,
        trend_strength, price_position, volume_ratio_regime,
        closing_prices[-1] - closing_prices[-2],  # Daily Return
        closing_prices[-1] - closing_prices[-3],  # 2-Day Return
        closing_prices[-1] - closing_prices[-5],  # 4-Day Return
        closing_prices[-1] - closing_prices[-10], # 9-Day Return
        np.mean(closing_prices[-5:]),             # Last 5-Day Average Price
        np.mean(closing_prices[-10:]),            # Last 10-Day Average Price
        np.mean(closing_prices[-20:]),            # Last 20-Day Average Price
        np.max(closing_prices[-20:]),              # Max Price in Last 20 Days
        np.min(closing_prices[-20:]),              # Min Price in Last 20 Days
        np.std(closing_prices[-20:]),               # Std Dev of Last 20 Days
        (closing_prices[-1] - np.mean(closing_prices[-5:])) / np.std(closing_prices[-5:]) if np.std(closing_prices[-5:]) != 0 else np.nan  # Z-score of Last Price
    ])

    # Create the enhanced state
    enhanced_s = np.concatenate((s, np.array(enhanced_features)), axis=0)
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    threshold = 2 * historical_volatility if historical_volatility > 0 else 0  # Adaptive threshold
    
    reward = 0
    
    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50  # Strong buy signal
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50  # Strong sell signal

    elif position_flag == 1:  # Holding
        if recent_return < -threshold:  # Sell signal
            reward -= 50  # Strong sell signal
        elif recent_return > 0:  # Positive return
            reward += 20  # Reward for holding during positive movement

    # Consider market conditions
    if np.isnan(threshold) or threshold < 1:  # Arbitrary threshold for "uncertain"
        reward -= 20  # Penalize for uncertain market conditions

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]