import numpy as np

def calculate_moving_averages(prices, windows):
    return {window: np.mean(prices[-window:]) for window in windows}

def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = np.abs(np.where(deltas < 0, deltas, 0)).mean()
    rs = gain / loss if loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-period:]) if len(tr) >= period else 0
    return atr

def calculate_obv(prices, volumes):
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif prices[i] < prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate moving averages
    sma_windows = [5, 10, 20]
    moving_averages = calculate_moving_averages(closing_prices, sma_windows)
    
    # Multi-timeframe Trend Indicators
    features = [
        moving_averages[5],
        moving_averages[10],
        moving_averages[20],
        closing_prices[-1] - moving_averages[5],
        closing_prices[-1] - moving_averages[10],
        closing_prices[-1] - moving_averages[20],
    ]
    
    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    features += [rsi_5, rsi_10, rsi_14]
    
    # Volatility Indicators
    historical_vol_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    historical_vol_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) * 100
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    features += [historical_vol_5, historical_vol_20, atr]
    
    # Volume-Price Relationship Indicators
    obv = calculate_obv(closing_prices, volumes)
    features += [obv[-1], np.corrcoef(closing_prices, volumes)[0, 1], volumes[-1] / np.mean(volumes)]
    
    # Market Regime Detection Indicators
    volatility_ratio = historical_vol_5 / historical_vol_20 if historical_vol_20 != 0 else 0
    trend_strength = np.corrcoef(np.arange(len(closing_prices)), closing_prices)[0, 1]**2
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes) if np.mean(volumes) != 0 else 0
    
    features += [volatility_ratio, trend_strength, price_position, volume_ratio_regime]

    # Combine features into a single state
    enhanced_s = np.concatenate([s, np.array(features)])
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Historical volatility
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    
    # Relative threshold based on volatility
    threshold = 2 * historical_vol
    
    reward = 0
    
    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong upward movement
            reward += 50
        elif recent_return < -threshold:  # Significant drop
            reward -= 30
    else:  # Holding
        if recent_return < -threshold:  # Significant drop
            reward -= 50
        elif recent_return > threshold:  # Strong upward movement
            reward += 30
    
    # Penalize uncertain market conditions
    volatility_ratio = enhanced_s[-5]  # 5-day volatility ratio
    if volatility_ratio > 2.0:  # Extreme volatility
        reward -= 20
    
    return np.clip(reward, -100, 100)