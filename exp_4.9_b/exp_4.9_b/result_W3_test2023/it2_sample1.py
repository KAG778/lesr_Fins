import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else np.nan

def calculate_obv(prices, volumes):
    if len(prices) < 2:
        return 0
    obv = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv += volumes[i]
        elif prices[i] < prices[i - 1]:
            obv -= volumes[i]
    return obv

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
    volumes = s[80:100]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    enhanced_features = []

    # Trend Indicators
    for window in [5, 10, 20]:
        enhanced_features.append(calculate_sma(closing_prices, window))  # SMA
        enhanced_features.append(calculate_ema(closing_prices, window))  # EMA

    # Momentum Indicators
    for window in [5, 10, 14]:
        enhanced_features.append(calculate_rsi(closing_prices, window))  # RSI

    # Daily Momentum
    enhanced_features.append(closing_prices[-1] - closing_prices[-2])  # Daily momentum

    # Volatility Indicators
    enhanced_features.append(np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 5 else np.nan)  # 5-day volatility
    enhanced_features.append(np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else np.nan)  # 20-day volatility
    enhanced_features.append(calculate_atr(high_prices, lows, closing_prices, 14))  # ATR

    # Volume Indicators
    obv = calculate_obv(closing_prices, volumes)  # On-Balance Volume
    enhanced_features.append(obv)
    enhanced_features.append(volumes[-1] / np.mean(volumes[-20:]))  # Volume ratio

    # Market Regime Detection
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)
    enhanced_features.append(np.std(closing_prices[-5:]) / np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else np.nan)  # Volatility Ratio
    trend_strength = linear_regression_r_squared(closing_prices)
    enhanced_features.append(trend_strength)  # R^2 of linear regression
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else 0
    enhanced_features.append(price_position)
    enhanced_features.append(np.mean(volumes[-5:]) / np.mean(volumes[-20:]))  # Volume Ratio Regime

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(enhanced_features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(daily_returns) if len(daily_returns) > 1 else 0
    threshold = 2 * historical_vol  # Adaptive threshold based on historical volatility

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong buy signal
            reward += 50
        elif recent_return < -threshold:  # Strong sell signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Significant drop
            reward -= 50
        elif recent_return > threshold:  # Positive return
            reward += 25
        if enhanced_s[-5] < 0.5:  # Weak trend signal
            reward -= 30  # Penalize holding in weak trend

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]