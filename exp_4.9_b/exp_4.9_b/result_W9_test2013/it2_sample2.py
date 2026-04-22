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
        sma = np.mean(closing_prices[-window:]) if len(closing_prices) >= window else np.nan
        new_features.append(sma)
        if window == 20:  # Price relative to 20-day SMA
            new_features.append(closing_prices[-1] - sma)
            new_features.append(closing_prices[-1] / sma)  # Price/SMA ratio

    # Exponential Moving Averages (EMA)
    ema_windows = [5, 10, 20, 50]
    for window in ema_windows:
        ema = np.mean(closing_prices[-window:]) if len(closing_prices) >= window else np.nan
        new_features.append(ema)
        if window == 20:  # Price relative to 20-day EMA
            new_features.append(closing_prices[-1] - ema)
            new_features.append(closing_prices[-1] / ema)  # Price/EMA ratio

    # B. Momentum Indicators
    rsi_windows = [5, 10, 14, 21]
    for period in rsi_windows:
        deltas = np.diff(closing_prices[-period:]) if len(closing_prices) >= period else np.array([])
        gains = np.where(deltas > 0, deltas, 0).mean() if len(deltas) > 0 else np.nan
        losses = np.abs(np.where(deltas < 0, deltas, 0)).mean() if len(deltas) > 0 else np.nan
        rs = gains / losses if losses > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        new_features.append(rsi)

    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 else np.nan
    new_features.append(momentum)

    # C. Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:]) / closing_prices[-5:-1]) * 100 if len(closing_prices) >= 5 else np.nan
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-20:-1]) * 100 if len(closing_prices) >= 20 else np.nan
    atr = np.mean(np.maximum(high_prices[-14:] - low_prices[-14:], 0)) if len(high_prices) >= 14 else np.nan
    new_features.extend([historical_volatility_5, historical_volatility_20, atr])

    # D. Volume-Price Relationship
    obv = np.sum(volumes[1:] * np.sign(np.diff(closing_prices))) if len(closing_prices) > 1 else np.nan
    volume_avg_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.nan
    volume_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan
    volume_ratio = volume_avg_5 / volume_avg_20 if volume_avg_20 > 0 else np.nan
    new_features.extend([obv, volume_avg_5, volume_avg_20, volume_ratio])

    # E. Market Regime Detection Indicators
    if historical_volatility_20 > 0:
        volatility_ratio = historical_volatility_5 / historical_volatility_20
    else:
        volatility_ratio = np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2 if len(closing_prices) > 1 else np.nan
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    new_features.extend([volatility_ratio, trend_strength, price_position, volume_ratio_regime])

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(new_features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Get the position flag (1.0 = holding, 0.0 = not holding)
    closing_prices = enhanced_s[0:20]

    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) >= 2 else 0

    # Historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Adaptive threshold based on historical volatility

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong upward movement
            reward += 50
        elif enhanced_s[120] < 30:  # Oversold condition
            reward += 20
        else:
            reward -= 10  # Uncertain market conditions

    else:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        elif recent_return > threshold:  # Strong upward movement
            reward += 30
        if enhanced_s[120] > 70:  # Overbought condition
            reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds