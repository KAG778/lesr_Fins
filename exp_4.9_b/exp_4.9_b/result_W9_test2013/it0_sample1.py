import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    return np.dot(weights, prices[-window:])

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    
    rs = gain / loss if loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    if np.isnan(ema12) or np.isnan(ema26):
        return np.nan, np.nan, np.nan
    
    macd_line = ema12 - ema26
    signal_line = calculate_ema(prices[-26:], 9)  # This would be based on the MACD line
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
    
    enhanced_s = []

    # Multi-timeframe Trend Indicators
    sma5 = calculate_sma(closing_prices, 5)
    sma10 = calculate_sma(closing_prices, 10)
    sma20 = calculate_sma(closing_prices, 20)
    ema5 = calculate_ema(closing_prices, 5)
    ema10 = calculate_ema(closing_prices, 10)
    
    enhanced_s.extend([sma5, sma10, sma20, ema5, ema10, closing_prices[-1] - sma20, closing_prices[-1] - ema10])

    # Momentum Indicators
    rsi5 = calculate_rsi(closing_prices, 5)
    rsi14 = calculate_rsi(closing_prices, 14)
    macd_line, signal_line, macd_hist = calculate_macd(closing_prices)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    
    enhanced_s.extend([rsi5, rsi14, macd_line, signal_line, macd_hist, momentum])

    # Volatility Indicators
    historical_vol5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 1 else np.nan
    historical_vol20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) * 100 if len(closing_prices) > 20 else np.nan
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    volatility_ratio = historical_vol5 / historical_vol20 if historical_vol20 else np.nan
    
    enhanced_s.extend([historical_vol5, historical_vol20, atr, volatility_ratio])

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))  # Simplified OBV
    avg_volume = np.mean(volumes)
    volume_ratio = volumes[-1] / avg_volume if avg_volume else np.nan
    
    enhanced_s.extend([obv[-1], np.corrcoef(closing_prices[:-1], volumes[1:])[0, 1], volume_ratio])

    # Market Regime Detection
    volatility_ratio_market = historical_vol5 / historical_vol20 if historical_vol20 else np.nan
    trend_strength = np.polyfit(range(len(closing_prices)), closing_prices, 1)[0]  # Linear regression slope
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes) if np.mean(volumes) else np.nan
    
    enhanced_s.extend([volatility_ratio_market, trend_strength, price_position, volume_ratio_regime])

    enhanced_s = np.array(enhanced_s)
    
    return np.concatenate((s, enhanced_s))

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    
    # Calculate threshold based on historical volatility
    threshold = 2 * historical_vol
    
    reward = 0
    
    if position == 0:  # Not holding
        if recent_return > threshold and enhanced_s[120] > 0:  # Assuming uptrend
            reward += 50  # Reward for buy signal
        elif recent_return < -threshold:
            reward -= 20  # Small penalty for downturn
    else:  # Holding
        if recent_return < -threshold:  # If price drops significantly
            reward -= 50  # Penalty for holding through downturn
        elif enhanced_s[120] > 0:  # Assuming uptrend
            reward += 10  # Small reward for holding

    return np.clip(reward, -100, 100)