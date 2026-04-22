import numpy as np

def revise_state(s):
    # Extract relevant price and volume data from the raw state
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Initialize an array for new features
    new_features = []

    # A. Multi-timeframe Trend Indicators
    sma_windows = [5, 10, 20, 50]
    for window in sma_windows:
        new_features.append(np.mean(closing_prices[-window:]) if len(closing_prices) >= window else np.nan)

    # Price vs Moving Averages
    for window in sma_windows:
        new_features.append(closing_prices[-1] - np.mean(closing_prices[-window:]) if len(closing_prices) >= window else np.nan)

    # B. Momentum Indicators
    rsi_windows = [5, 10, 14, 21]
    for window in rsi_windows:
        deltas = np.diff(closing_prices[-window:]) if len(closing_prices) >= window else []
        gains = np.where(deltas > 0, deltas, 0).mean() if deltas.size > 0 else np.nan
        losses = -np.where(deltas < 0, deltas, 0).mean() if deltas.size > 0 else np.nan
        rs = gains / losses if losses != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        new_features.append(rsi)

    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 else np.nan
    new_features.append(momentum)

    # C. Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 5 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else 0
    atr = np.mean(np.maximum(high_prices[-14:] - low_prices[-14:], 0)) if len(high_prices) >= 14 else np.nan
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else np.nan

    new_features.extend([historical_volatility_5, historical_volatility_20, atr, volatility_ratio])

    # D. Volume-Price Relationship
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:])) if len(closing_prices) > 1 else 0
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan

    new_features.extend([obv, volume_ratio])

    # E. Market Regime Detection Indicators
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2 if len(closing_prices) > 1 else 0
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else 0

    new_features.extend([volatility_ratio, trend_strength, price_position, volume_ratio_regime])

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(new_features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Get the position flag (1.0 = holding, 0.0 = not holding)
    closing_prices = enhanced_s[0:20]
    
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) >= 2 else 0
    historical_vol = np.std(np.diff(closing_prices)) * 100 if len(closing_prices) > 1 else 0  # Convert to percentage
    threshold = 2 * historical_vol  # Adaptive threshold based on historical volatility

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong upward movement
            reward += 50
        elif enhanced_s[120] < 30:  # Oversold condition (RSI)
            reward += 20
        else:
            reward -= 10  # Penalize uncertain market conditions

    else:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        elif recent_return > threshold:  # Strong upward movement
            reward += 30
        if enhanced_s[120] > 70:  # Overbought condition (RSI)
            reward -= 20  # Penalize for overbought

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds