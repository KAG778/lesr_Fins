import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])  # Start with SMA for the first value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema[-1]  # Return the last EMA value

def calculate_rsi(prices, period):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = np.abs(np.where(deltas < 0, deltas, 0)).mean()
    rs = gain / loss if loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period):
    """Calculate Average True Range."""
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-period:]) if len(tr) >= period else np.nan

def calculate_obv(prices, volumes):
    """Calculate On-Balance Volume."""
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif prices[i] < prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv[-1]  # Return the last OBV value

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
    # Moving Averages
    new_features.extend([calculate_sma(closing_prices, window) for window in [5, 10, 20, 50]])
    new_features.extend([calculate_ema(closing_prices, window) for window in [5, 10, 20]])

    # Price vs Moving Averages
    new_features.append(closing_prices[-1] - calculate_sma(closing_prices, 20))  # Price - SMA(20)
    new_features.append(closing_prices[-1] - calculate_sma(closing_prices, 50))  # Price - SMA(50)

    # B. Momentum Indicators
    new_features.append(calculate_rsi(closing_prices, 5))
    new_features.append(calculate_rsi(closing_prices, 10))
    new_features.append(calculate_rsi(closing_prices, 14))
    
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]  # Rate of change
    new_features.append(momentum)

    # C. Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 5 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    new_features.append(historical_volatility_5)
    new_features.append(historical_volatility_20)
    new_features.append(atr)

    # Volatility Ratio
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else 0
    new_features.append(volatility_ratio)

    # D. Volume-Price Relationship Indicators
    obv = calculate_obv(closing_prices, volumes)
    new_features.append(obv)

    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else 0
    new_features.append(volume_ratio)

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

    # Historical volatility
    historical_vol = np.std(np.diff(closing_prices)) if len(closing_prices) > 1 else 0
    threshold = 2 * historical_vol  # Adaptive threshold based on historical volatility

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong upward movement
            reward += 50
        elif enhanced_s[120] < 30:  # Oversold condition (RSI)
            reward += 20
        else:
            reward -= 10  # Uncertain market conditions

    else:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        elif recent_return > threshold:  # Strong upward movement
            reward += 30
        if enhanced_s[120] > 70:  # Overbought condition (RSI)
            reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds