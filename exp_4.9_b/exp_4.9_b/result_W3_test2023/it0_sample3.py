import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window), 'valid') / window

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else 0

def linear_regression_r_squared(prices):
    n = len(prices)
    if n < 2:
        return 0
    x = np.arange(n)
    y = prices
    m = np.vstack([x, np.ones(n)]).T
    coeffs = np.linalg.lstsq(m, y, rcond=None)[0]
    predictions = m @ coeffs
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adj_closing_prices = s[100:120]

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else 0
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else 0
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else 0
    price_sma_5_diff = closing_prices[-1] - sma_5
    price_sma_10_diff = closing_prices[-1] - sma_10
    price_sma_20_diff = closing_prices[-1] - sma_20

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # MACD
    ema_12 = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    ema_26 = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    macd_line = ema_12 - ema_26
    signal_line = np.mean(closing_prices[-9:]) if len(closing_prices) >= 9 else 0  # EMA of MACD
    macd_histogram = macd_line - signal_line

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 5 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume-Price Relationship
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], np.where(np.diff(closing_prices) < 0, -volumes[1:], 0)))
    volume_avg_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    volume_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    volume_ratio = volume_avg_5 / volume_avg_20 if volume_avg_20 != 0 else 0

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else 0
    trend_strength = linear_regression_r_squared(closing_prices)
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else 0
    volume_ratio_regime = volume_avg_5 / volume_avg_20 if volume_avg_20 != 0 else 0

    enhanced_s = np.concatenate((
        s,
        [sma_5, sma_10, sma_20, price_sma_5_diff, price_sma_10_diff, price_sma_20_diff,
         rsi_5, rsi_10, rsi_14, macd_line, signal_line, macd_histogram,
         historical_volatility_5, historical_volatility_20, atr, obv, volume_ratio,
         volatility_ratio, trend_strength, price_position, volume_ratio_regime]
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    position_flag = enhanced_s[-1]

    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices)) if len(closing_prices) > 1 else 0
    threshold = 2 * historical_vol  # Adaptive threshold based on volatility

    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold:
            reward += 50  # Strong BUY signal
        elif recent_return < -threshold:
            reward -= 50  # Weak signal
    else:  # Holding
        if recent_return < -threshold:
            reward -= 50  # Weak signal
        elif recent_return > threshold:
            reward += 25  # Positive reward for holding during uptrend

    return np.clip(reward, -100, 100)