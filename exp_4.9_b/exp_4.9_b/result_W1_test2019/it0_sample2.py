import numpy as np

def calculate_moving_averages(prices, windows):
    return {window: np.convolve(prices, np.ones(window) / window, mode='valid') for window in windows}

def calculate_rsi(prices, periods):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi_values = {}
    for period in periods:
        avg_gain = np.convolve(gains, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period) / period, mode='valid')
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_values[period] = np.concatenate([np.full(period - 1, np.nan), rsi])
    
    return rsi_values

def calculate_atr(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:], 
                   np.maximum(np.abs(highs[1:] - closes[:-1]), 
                              np.abs(lows[1:] - closes[:-1])))
    atr = np.convolve(tr, np.ones(period) / period, mode='valid')
    return np.concatenate([np.full(period - 1, np.nan), atr])

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Multi-timeframe Trend Indicators
    ma_windows = [5, 10, 20]
    moving_averages = calculate_moving_averages(closing_prices, ma_windows)
    features = []
    
    for window in ma_windows:
        features.append(closing_prices[-1] / moving_averages[window][-1])  # Price relative to MA
        if window > 5:
            features.append(moving_averages[5][-1] - moving_averages[window][-1])  # Short vs Long MA difference

    # Momentum Indicators
    rsi_values = calculate_rsi(closing_prices, [5, 10, 14])
    for period in [5, 10, 14]:
        features.append(rsi_values[period][-1])

    # MACD
    ema12 = np.convolve(closing_prices, np.ones(12) / 12, mode='valid')
    ema26 = np.convolve(closing_prices, np.ones(26) / 26, mode='valid')
    macd = ema12[-len(ema26):] - ema26
    signal_line = np.convolve(macd, np.ones(9) / 9, mode='valid')
    features.append(macd[-1])  # MACD line
    features.append(signal_line[-1])  # Signal line
    features.append(macd[-1] - signal_line[-1])  # MACD histogram

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) * 100 if len(closing_prices) > 20 else np.nan
    features.append(historical_volatility_5)
    features.append(historical_volatility_20)

    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    features.append(atr[-1])  # ATR

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    features.append(obv[-1])  # OBV
    features.append(volumes[-1] / np.mean(volumes[-20:]))  # Volume ratio

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 > 0 else np.nan
    trend_strength = np.corrcoef(np.arange(len(closing_prices)), closing_prices)[0, 1] ** 2  # R²
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    features.append(volatility_ratio)
    features.append(trend_strength)
    features.append(price_position)
    features.append(volume_ratio_regime)

    # Combine all features into the enhanced state
    enhanced_s = np.concatenate((s, np.array(features)))
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100

    threshold = 2 * historical_vol  # Volatility-adaptive threshold
    reward = 0

    if position_flag == 0:  # Not holding
        if enhanced_s[-5] > 70:  # RSI 14
            reward += 10  # Potential BUY signal for oversold
        if enhanced_s[-6] > 0.1:  # Trend strength
            reward += 10  # Strong uptrend signal
    else:  # Holding
        if recent_return < -threshold:
            reward -= 50  # Significant drop
        elif enhanced_s[-5] < 30:  # RSI 14
            reward -= 10  # Potential SELL signal for overbought
        if enhanced_s[-6] < 0.1:  # Trend strength
            reward -= 10  # Weak trend signal

    return np.clip(reward, -100, 100)  # Limit reward to range [-100, 100]