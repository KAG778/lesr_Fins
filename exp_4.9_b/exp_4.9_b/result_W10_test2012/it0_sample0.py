import numpy as np

def moving_average(data, window):
    return np.convolve(data, np.ones(window), 'valid') / window

def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    short_ema = np.mean(prices[-12:])  # 12-day EMA
    long_ema = np.mean(prices[-26:])   # 26-day EMA
    macd_line = short_ema - long_ema
    signal_line = np.mean(prices[-9:])  # 9-day EMA of MACD
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(highs, lows, closes, period=14):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-period:]) if len(tr) >= period else 0
    return atr

def revise_state(raw_state):
    closing_prices = raw_state[0:20]
    opening_prices = raw_state[20:40]
    high_prices = raw_state[40:60]
    low_prices = raw_state[60:80]
    volumes = raw_state[80:100]

    # Calculate moving averages
    sma_5 = moving_average(closing_prices, 5)
    sma_10 = moving_average(closing_prices, 10)
    sma_20 = moving_average(closing_prices, 20)

    # Trend indicators
    trend_5_20_diff = sma_5[-1] - sma_20[-1] if len(sma_5) > 0 and len(sma_20) > 0 else 0
    price_relative_to_ma_5 = closing_prices[-1] - sma_5[-1] if len(sma_5) > 0 else 0
    price_relative_to_ma_10 = closing_prices[-1] - sma_10[-1] if len(sma_10) > 0 else 0

    # Momentum indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    macd_line, signal_line, histogram = calculate_macd(closing_prices)

    # Volatility indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:]) / closing_prices[-5:-1]) * 100 if len(closing_prices) > 5 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-20:-1]) * 100 if len(closing_prices) > 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices)
    
    # Volume-price relationship
    obv = np.cumsum(np.sign(np.diff(closing_prices)) * volumes[1:])
    volume_ratio = volumes[-5:].mean() / volumes[-20:].mean() if len(volumes) > 20 else 0

    # Market regime detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else 0
    trend_strength = np.corrcoef(np.arange(len(closing_prices)), closing_prices)[0, 1] ** 2  # R²
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else 0
    volume_ratio_regime = volumes[-5:].mean() / volumes[-20:].mean() if len(volumes) > 20 else 0

    # Assemble enhanced state
    enhanced_s = np.concatenate((
        raw_state,
        np.array([
            sma_5[-1] if len(sma_5) > 0 else 0,
            sma_10[-1] if len(sma_10) > 0 else 0,
            sma_20[-1] if len(sma_20) > 0 else 0,
            trend_5_20_diff,
            price_relative_to_ma_5,
            price_relative_to_ma_10,
            rsi_5,
            rsi_10,
            rsi_14,
            macd_line,
            signal_line,
            histogram,
            historical_volatility_5,
            historical_volatility_20,
            atr,
            obv[-1] if len(obv) > 0 else 0,
            volume_ratio,
            volatility_ratio,
            trend_strength,
            price_position,
            volume_ratio_regime,
        ])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 1 else 0
    threshold = 2 * historical_volatility  # Adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong BUY signal
            reward += 50
        elif recent_return < -threshold:  # Avoid buying in a downturn
            reward -= 50
    else:  # Holding
        if recent_return > threshold:  # Good to HOLD
            reward += 30
        elif recent_return < -threshold:  # Sell signal in a downturn
            reward -= 50

    return reward