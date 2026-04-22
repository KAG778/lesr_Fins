import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])  # Start with SMA for the first value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema[-1]  # Return the last EMA value

def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = np.abs(np.where(deltas < 0, deltas, 0)).mean()
    rs = gain / loss if loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-period:]) if len(tr) >= period else np.nan

def calculate_obv(prices, volumes):
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
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # Volatility Indicators
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 5 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else 0
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else 0

    # Volume-Price Relationship
    obv = calculate_obv(closing_prices, volumes)
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else 0

    # Market Regime Detection
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else 0

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,
        [sma_5, sma_10, sma_20, ema_5, ema_10, rsi_5, rsi_10, rsi_14, atr,
         historical_volatility_5, historical_volatility_20, volatility_ratio, obv, volume_ratio,
         trend_strength, price_position]
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Get the position flag (1.0 = holding, 0.0 = not holding)
    closing_prices = enhanced_s[0:20]
    historical_vol = np.std(np.diff(closing_prices)) if len(closing_prices) > 1 else 0
    threshold = 2 * historical_vol  # Use 2x historical volatility as threshold

    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong upward movement
            reward += 50
        elif recent_return < -threshold:  # Significant drop
            reward -= 30
        # Additional indicators for buy
        if enhanced_s[120] < 30:  # Oversold condition (RSI)
            reward += 20

    else:  # Holding
        if recent_return < -threshold:  # Significant drop
            reward -= 50
        elif recent_return > threshold:  # Strong upward movement
            reward += 30
        # Additional indicators for sell
        if enhanced_s[120] > 70:  # Overbought condition (RSI)
            reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds