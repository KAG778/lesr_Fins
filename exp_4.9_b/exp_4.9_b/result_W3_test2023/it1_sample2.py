import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss > 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else 0

def calculate_historical_volatility(prices, window):
    returns = np.diff(prices) / prices[:-1] * 100
    return np.std(returns) if len(returns) >= 2 else np.nan

def calculate_obv(prices, volumes):
    if len(prices) < 2:
        return 0
    obv = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv += volumes[i]
        elif prices[i] < prices[i - 1]:
            obv -= volumes[i]
    return obv

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    enhanced_features = []
   
    # Trend Indicators
    enhanced_features.append(calculate_sma(closing_prices, 5))
    enhanced_features.append(calculate_sma(closing_prices, 10))
    enhanced_features.append(calculate_sma(closing_prices, 20))
    enhanced_features.append(calculate_ema(closing_prices, 5))
    enhanced_features.append(calculate_ema(closing_prices, 10))
    
    # Momentum Indicators
    enhanced_features.append(calculate_rsi(closing_prices, 5))
    enhanced_features.append(calculate_rsi(closing_prices, 14))
    
    # Volatility Indicators
    enhanced_features.append(calculate_historical_volatility(closing_prices, 5))
    enhanced_features.append(calculate_historical_volatility(closing_prices, 20))
    enhanced_features.append(calculate_atr(high_prices, lows, closing_prices, 14))
    
    # Volume-Price Relationship
    obv = calculate_obv(closing_prices, volumes)
    enhanced_features.append(obv)
    enhanced_features.append(volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0)  # Volume ratio

    # Market Regime Detection
    volatility_ratio = enhanced_features[-2] / enhanced_features[-3] if enhanced_features[-3] > 0 else 0  # Historical vol ratio
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]  # Linear regression R²
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else 0
    
    enhanced_features.append(volatility_ratio)
    enhanced_features.append(trend_strength)
    enhanced_features.append(price_position)
    enhanced_features.append(volume_ratio_regime)

    # Combine original state and new features
    enhanced_s = np.concatenate((s, np.array(enhanced_features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    historical_vol = calculate_historical_volatility(closing_prices, 20)
    threshold = 2 * historical_vol if historical_vol > 0 else 0
    
    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50
    else:  # Holding
        if recent_return > threshold:  # Positive return
            reward += 25
        if recent_return < -threshold:  # Significant drop
            reward -= 50
        if enhanced_s[-3] < 0.5:  # Weak trend signal
            reward -= 50  # Penalize for holding in weak conditions

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]