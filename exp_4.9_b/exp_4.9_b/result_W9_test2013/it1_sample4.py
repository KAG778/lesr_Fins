import numpy as np

def moving_average(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0).mean()
    losses = -np.where(deltas < 0, deltas, 0).mean()
    rs = gains / losses if losses != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-period:]) if len(tr) >= period else 0
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Initialize new features array
    new_features = []
    
    # A. Trend Indicators
    new_features.append(moving_average(closing_prices, 5))
    new_features.append(moving_average(closing_prices, 10))
    new_features.append(moving_average(closing_prices, 20))
    new_features.append(moving_average(closing_prices, 50))
    
    # Price vs Moving Averages
    new_features.append(closing_prices[-1] - moving_average(closing_prices, 20))
    new_features.append(closing_prices[-1] - moving_average(closing_prices, 50))

    # B. Momentum Indicators
    new_features.append(calculate_rsi(closing_prices, 5))
    new_features.append(calculate_rsi(closing_prices, 14))
    new_features.append(calculate_rsi(closing_prices, 21))
    
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]  # Rate of change
    new_features.append(momentum)

    # C. Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 1 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) * 100 if len(closing_prices) > 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    new_features.append(historical_volatility_5)
    new_features.append(historical_volatility_20)
    new_features.append(atr)
    
    # D. Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) != 0 else 0
    new_features.append(obv[-1] if len(obv) > 0 else 0)
    new_features.append(volume_ratio)

    # E. Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2  # R² of linear regression
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else 0
    
    new_features.append(volatility_ratio)
    new_features.append(trend_strength)
    new_features.append(price_position)
    new_features.append(volume_ratio_regime)

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(new_features)))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Convert to percentage
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100  # Daily volatility percentage
    threshold = 2 * historical_vol  # Adaptive threshold based on volatility

    reward = 0
    
    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong upward movement
            reward += 50
        elif enhanced_s[120] < 30:  # Oversold condition
            reward += 20
        else:
            reward -= 10  # Uncertain market conditions

    elif position == 1:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        elif enhanced_s[120] > 70:  # Overbought condition
            reward += 20
        else:
            reward += 10  # Maintain position in uptrend
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds