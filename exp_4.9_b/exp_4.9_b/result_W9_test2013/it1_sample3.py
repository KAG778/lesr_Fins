import numpy as np

def calculate_sma(prices, period):
    return np.mean(prices[-period:]) if len(prices) >= period else np.nan

def calculate_ema(prices, period):
    if len(prices) < period:
        return np.nan
    weights = np.exp(np.linspace(-1, 0, period))
    weights /= weights.sum()
    return np.dot(weights, prices[-period:])

def calculate_rsi(prices, period):
    if len(prices) < period:
        return np.nan
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0).mean()
    losses = -np.where(deltas < 0, deltas, 0).mean()
    rs = gains / losses if losses > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period):
    if len(highs) < period:
        return np.nan
    tr = np.maximum(highs[-period:] - lows[-period:], 
                    np.maximum(np.abs(highs[-period:] - closes[-period:-1]), 
                               np.abs(lows[-period:] - closes[-period:-1])))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    new_features = []
    
    # Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    new_features.extend([sma_5, sma_10, sma_20, ema_5, ema_10])
    
    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    
    new_features.extend([rsi_5, rsi_14, momentum])

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices)) if len(closing_prices) > 1 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    new_features.extend([historical_volatility_5, historical_volatility_20, atr])

    # Volume-Price Relationship
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0
    
    new_features.extend([obv, volume_ratio])

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes) if np.mean(volumes) != 0 else 0
    
    new_features.extend([volatility_ratio, trend_strength, price_position, volume_ratio_regime])
    
    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(new_features)))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices)) * 100  # Convert to percentage
    
    # Use volatility-adaptive thresholds
    threshold = 2 * historical_vol if historical_vol > 0 else 1  # Avoid division by zero
    
    reward = 0
    
    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong upward movement
            reward += 50  # Reward for buy signal
        elif recent_return < -threshold:  # Significant drop
            reward -= 20  # Small penalty for downturn
        else:
            reward -= 10  # Penalize uncertain conditions

    else:  # Holding
        if recent_return < -threshold:  # Significant drop
            reward -= 50  # Penalty for holding through downturn
        elif recent_return > threshold:  # Strong upward movement
            reward += 30  # Small reward for holding
        else:
            reward -= 10  # Penalize if returns are neutral

    # Penalize extreme volatility
    volatility_ratio = enhanced_s[-5]  # 5-day volatility ratio
    if volatility_ratio > 2.0:  # Extreme volatility
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds