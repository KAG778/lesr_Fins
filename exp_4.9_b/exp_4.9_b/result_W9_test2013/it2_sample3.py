import numpy as np

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Initialize new features array
    new_features = []

    # A. Multi-timeframe Trend Indicators
    sma_5 = np.mean(closing_prices[-5:])
    sma_10 = np.mean(closing_prices[-10:])
    sma_20 = np.mean(closing_prices[-20:])
    sma_50 = np.mean(closing_prices[-50:]) if len(closing_prices) >= 50 else np.nan
    ema_5 = np.mean(closing_prices[-5:])  # Placeholder for EMA (can be refined)
    ema_10 = np.mean(closing_prices[-10:])  # Placeholder for EMA
    ema_20 = np.mean(closing_prices[-20:])  # Placeholder for EMA
    price_above_sma_20 = closing_prices[-1] > sma_20
    
    new_features.extend([
        sma_5, sma_10, sma_20, sma_50,
        ema_5, ema_10,
        closing_prices[-1] - sma_10,
        closing_prices[-1] - sma_20,
        closing_prices[-1] - sma_50,
        price_above_sma_20
    ])
    
    # B. Momentum Indicators
    def rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.abs(np.where(deltas < 0, deltas, 0)).mean()
        rs = gains / losses if losses != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi_5 = rsi(closing_prices[-5:])
    rsi_10 = rsi(closing_prices[-10:])
    rsi_14 = rsi(closing_prices[-14:]) if len(closing_prices) >= 14 else 0
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]  # Rate of change

    new_features.extend([rsi_5, rsi_10, rsi_14, momentum])

    # C. Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 5 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else 0
    historical_volatility_50 = np.std(np.diff(closing_prices[-50:])) if len(closing_prices) >= 50 else 0
    atr = np.mean(np.maximum(high_prices[-20:] - low_prices[-20:], 0)) if len(high_prices) >= 20 else 0
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else 0

    new_features.extend([
        historical_volatility_5, historical_volatility_20, historical_volatility_50, atr, volatility_ratio
    ])

    # D. Volume-Price Relationship Indicators
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))  # On-Balance Volume
    volume_avg_5 = np.mean(volumes[-5:])
    volume_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    volume_ratio = volume_avg_5 / volume_avg_20 if volume_avg_20 != 0 else 0

    new_features.extend([obv, volume_avg_5, volume_avg_20, volume_ratio])

    # E. Market Regime Detection Indicators
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2  # R²
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else 0

    new_features.extend([trend_strength, price_position, volume_ratio_regime])

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(new_features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Get the position flag (1.0 = holding, 0.0 = not holding)
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Convert to percentage
    historical_vol = np.std(np.diff(closing_prices)) * 100 if len(closing_prices) > 1 else 0  # Daily volatility percentage
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
        elif recent_return > threshold:  # Strong upward movement
            reward += 30
        # Penalize for extreme volatility
        if enhanced_s[5] > 2.0:  # 5-day volatility ratio
            reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds