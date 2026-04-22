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
    sma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    sma_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan
    sma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else np.nan
    sma_50 = np.mean(closing_prices[-50:]) if len(closing_prices) >= 50 else np.nan
    ema_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    ema_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan
    price_above_sma_20 = closing_prices[-1] > sma_20 if sma_20 is not np.nan else np.nan
    price_above_sma_50 = closing_prices[-1] > sma_50 if sma_50 is not np.nan else np.nan

    new_features.extend([sma_5, sma_10, sma_20, sma_50, ema_5, ema_10, price_above_sma_20, price_above_sma_50])

    # B. Momentum Indicators
    def rsi(prices, period):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.abs(np.where(deltas < 0, deltas, 0)).mean()
        rs = gains / losses if losses != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi_5 = rsi(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    rsi_10 = rsi(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan
    rsi_14 = rsi(closing_prices[-14:]) if len(closing_prices) >= 14 else np.nan
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 else np.nan

    new_features.extend([rsi_5, rsi_10, rsi_14, momentum])

    # C. Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:])) * 100 if len(closing_prices) >= 5 else np.nan
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) * 100 if len(closing_prices) >= 20 else np.nan
    historical_volatility_50 = np.std(np.diff(closing_prices[-50:])) * 100 if len(closing_prices) >= 50 else np.nan
    atr = np.mean(high_prices[-14:] - low_prices[-14:]) if len(high_prices) >= 14 else np.nan  # Average True Range

    volatility_ratio_5_20 = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else np.nan
    volatility_ratio_5_50 = historical_volatility_5 / historical_volatility_50 if historical_volatility_50 != 0 else np.nan

    new_features.extend([historical_volatility_5, historical_volatility_20, historical_volatility_50, atr, volatility_ratio_5_20, volatility_ratio_5_50])

    # D. Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))[-1] if len(closing_prices) > 1 else 0
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.nan
    volume_ratio = volumes[-1] / avg_volume if avg_volume else np.nan

    new_features.extend([obv, volume_ratio])

    # E. Market Regime Detection
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2 if len(closing_prices) > 1 else np.nan
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if len(closing_prices) >= 20 and np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan

    new_features.extend([trend_strength, price_position, volume_ratio_regime])

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(new_features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Get the position flag (1.0 = holding, 0.0 = not holding)
    closing_prices = enhanced_s[0:20]

    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0

    # Calculate historical volatility for adaptive threshold
    historical_vol = np.std(np.diff(closing_prices)) * 100 if len(closing_prices) > 1 else 0
    threshold = 2 * historical_vol  # Use 2x historical volatility as threshold

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong upward movement
            reward += 50
        elif enhanced_s[120] < 30:  # Oversold condition (RSI example)
            reward += 20
        else:
            reward -= 10  # Penalize uncertain market conditions

    elif position == 1:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        elif recent_return > threshold:  # Strong upward movement
            reward += 30
        if enhanced_s[120] > 70:  # Overbought condition (RSI example)
            reward -= 20  # Penalize holding in overbought conditions

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds