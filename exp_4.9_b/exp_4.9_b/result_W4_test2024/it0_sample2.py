import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = [prices[0]]  # Start with the first price
    for price in prices[1:]:
        ema.append((price - ema[-1]) * alpha + ema[-1])
    return np.array(ema)

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-window:]) if len(tr) >= window else np.nan
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else np.nan
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    ema_10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    price_vs_sma_20 = closing_prices[-1] / sma_20 if sma_20 else np.nan

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5) if len(closing_prices) >= 5 else np.nan
    rsi_14 = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) >= 2 else np.nan

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices)) if len(closing_prices) >= 5 else np.nan
    historical_volatility_20 = np.std(np.diff(closing_prices)) if len(closing_prices) >= 20 else np.nan
    
    # Volume-Price Relationship
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], 
                          np.where(np.diff(closing_prices) < 0, -volumes[1:], 0)))

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 > 0 else np.nan
    trend_strength = np.nan  # Placeholder for linear regression R² calculation
    price_position = (closing_prices[-1] - low_prices[-20]) / (high_prices[-20] - low_prices[-20]) if high_prices[-20] != low_prices[-20] else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Compile the new features
    new_features = np.array([
        sma_5, sma_10, sma_20, ema_5, ema_10, price_vs_sma_20,
        rsi_5, rsi_14, atr_14, momentum,
        historical_volatility_5, historical_volatility_20,
        obv,
        volatility_ratio, trend_strength, price_position, volume_ratio_regime
    ])

    return np.concatenate((s, new_features))

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) >= 2 else 0
    historical_vol = np.std(np.diff(closing_prices)) if len(closing_prices) >= 5 else 1  # Avoid division by zero
    
    # Use 2x historical volatility as threshold for rewards
    threshold = 2 * historical_vol

    reward = 0
    if position_flag == 0.0:  # Not holding
        if recent_return > threshold:  # Strong uptrend signal
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend signal
            reward -= 50
    else:  # Holding
        if recent_return > 0:  # Positive return
            reward += 10
        elif recent_return < -threshold:  # Weakening trend
            reward -= 50

    return np.clip(reward, -100, 100)