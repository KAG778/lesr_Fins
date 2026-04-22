import numpy as np

def calculate_sma(data, window):
    if len(data) < window:
        return np.nan
    return np.mean(data[-window:])

def calculate_ema(data, window):
    if len(data) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros_like(data)
    ema[window-1] = np.mean(data[:window])
    for i in range(window, len(data)):
        ema[i] = (data[i] - ema[i-1]) * alpha + ema[i-1]
    return ema[-1]

def calculate_rsi(data, window):
    if len(data) < window:
        return np.nan
    deltas = np.diff(data)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, window):
    if len(high) < window:
        return np.nan
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.mean(tr[-window:])
    return atr

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * 100  # Convert to percentage

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]
    
    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    price_vs_sma_5 = closing_prices[-1] / sma_5 if sma_5 else np.nan
    price_vs_sma_20 = closing_prices[-1] / sma_20 if sma_20 else np.nan

    trend_diff = sma_5 - sma_20 if sma_20 else np.nan

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    macd = ema_5 - ema_10 if ema_10 else np.nan  # Simplified MACD
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Volatility Indicators
    historical_volatility_5 = calculate_volatility(closing_prices, 5)
    historical_volatility_20 = calculate_volatility(closing_prices, 20)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume-Price Relationship
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    avg_volume_5 = np.mean(volumes[-5:])
    avg_volume_20 = np.mean(volumes[-20:])
    volume_ratio = avg_volume_5 / avg_volume_20 if avg_volume_20 else np.nan

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 else np.nan
    trend_strength = np.nan  # Placeholder, to be calculated with regression R²
    price_position = (closing_prices[-1] - min(closing_prices[-20:])) / (max(closing_prices[-20:]) - min(closing_prices[-20:])) if max(closing_prices[-20:]) != min(closing_prices[-20:]) else 0
    volume_ratio_regime = avg_volume_5 / avg_volume_20 if avg_volume_20 else np.nan

    # Assemble enhanced state
    enhanced_s = np.concatenate([
        s,
        [sma_5, sma_10, sma_20, ema_5, ema_10, 
         price_vs_sma_5, price_vs_sma_20, trend_diff,
         rsi_5, rsi_10, macd, momentum,
         historical_volatility_5, historical_volatility_20, atr,
         obv, volume_ratio,
         volatility_ratio, trend_strength, price_position, volume_ratio_regime]
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    recent_return = (enhanced_s[0] - enhanced_s[1]) / enhanced_s[1] * 100  # Daily return calculation
    historical_volatility = np.std(np.diff(enhanced_s[:20]) / enhanced_s[:19]) * 100  # Historical volatility from closing prices

    threshold = 2 * historical_volatility  # Volatility-adaptive threshold

    reward = 0
    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Trend weakens
            reward -= 50
        elif recent_return > 0:  # Positive return
            reward += 30
        else:  # Choppy market
            reward -= 10

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]