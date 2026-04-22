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
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * (prices[i] - ema[i-1]) + ema[i-1]
    return ema[-1]

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    if len(closes) < window:
        return np.nan
    tr = np.maximum(highs[-1] - lows[-1], np.maximum(abs(highs[-1] - closes[-2]), abs(lows[-1] - closes[-2])))
    atr = np.mean(tr)  # Simplified ATR calculation
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Multi-timeframe Trend Indicators
    sma5 = calculate_sma(closing_prices, 5)
    sma10 = calculate_sma(closing_prices, 10)
    sma20 = calculate_sma(closing_prices, 20)
    ema5 = calculate_ema(closing_prices, 5)
    ema10 = calculate_ema(closing_prices, 10)
    
    trend_sma_diff = sma5 - sma20
    price_sma_ratio = closing_prices[-1] / sma20 if sma20 != 0 else np.nan
    
    # Momentum Indicators
    rsi5 = calculate_rsi(closing_prices, 5)
    rsi14 = calculate_rsi(closing_prices, 14)
    momentum = closing_prices[-1] - closing_prices[-2]
    
    # Volatility Indicators
    hist_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) if len(closing_prices) > 1 else np.nan
    hist_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) if len(closing_prices) > 20 else np.nan
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume-Price Relationship
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan
    
    # Market Regime Detection
    volatility_ratio = hist_volatility_5 / hist_volatility_20 if hist_volatility_20 != 0 else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0][1]  # Simplified R^2 calculation
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan

    # Collecting all features
    enhanced_s = np.array([
        sma5, sma10, sma20, ema5, ema10, trend_sma_diff, price_sma_ratio,
        rsi5, rsi14, momentum,
        hist_volatility_5, hist_volatility_20, atr,
        obv, volume_ratio,
        volatility_ratio, trend_strength, price_position, volume_ratio_regime
    ])

    return np.concatenate((s, enhanced_s))

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag: 1.0 = holding, 0.0 = not holding
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Volatility adaptive threshold

    reward = 0
    
    if position == 0:  # Not holding
        if recent_return > threshold:
            reward += 50  # Strong uptrend
        elif recent_return < -threshold:
            reward -= 50  # Strong downtrend
    else:  # Holding
        if recent_return > 0:
            reward += 20  # Holding during uptrend
        elif recent_return < -threshold:
            reward -= 50  # Weakening trend, consider selling
            
    return np.clip(reward, -100, 100)