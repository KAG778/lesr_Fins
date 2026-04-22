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

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    return np.std(np.diff(prices) / prices[:-1] * 100)

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
    
    enhanced_features = []

    # Multi-timeframe Trend Indicators
    enhanced_features.append(calculate_sma(closing_prices, 5))  # 5-day SMA
    enhanced_features.append(calculate_sma(closing_prices, 10))  # 10-day SMA
    enhanced_features.append(calculate_sma(closing_prices, 20))  # 20-day SMA
    enhanced_features.append(calculate_ema(closing_prices, 5))  # 5-day EMA
    enhanced_features.append(calculate_ema(closing_prices, 10))  # 10-day EMA
    enhanced_features.append(calculate_ema(closing_prices, 20))  # 20-day EMA

    # Momentum Indicators
    enhanced_features.append(calculate_rsi(closing_prices, 5))  # 5-day RSI
    enhanced_features.append(calculate_rsi(closing_prices, 14))  # 14-day RSI
    enhanced_features.append(closing_prices[-1] - closing_prices[-2])  # Daily momentum

    # Volatility Indicators
    enhanced_features.append(calculate_volatility(closing_prices, 5))  # 5-day volatility
    enhanced_features.append(calculate_volatility(closing_prices, 20))  # 20-day volatility

    # ATR
    high_prices = s[40:60]
    low_prices = s[60:80]
    atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))[:14])  # ATR for 14 days
    enhanced_features.append(atr)

    # Volume Indicators
    obv = calculate_obv(closing_prices, volumes)  # On-Balance Volume
    enhanced_features.append(obv)
    enhanced_features.append(np.mean(volumes[-5:]) / np.mean(volumes[-20:]))  # Volume ratio

    # Market Regime Detection
    volatility_ratio = calculate_volatility(closing_prices, 5) / calculate_volatility(closing_prices, 20) if len(closing_prices) >= 20 else np.nan
    trend_strength = linear_regression_r_squared(closing_prices)
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else 0

    enhanced_features.append(volatility_ratio)
    enhanced_features.append(trend_strength)
    enhanced_features.append(price_position)

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

    # Use adaptive threshold based on historical volatility
    threshold = 2 * historical_vol  # 2x standard deviation for risk management

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong buy signal
            reward += 50
        elif recent_return < -threshold:  # Strong sell signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        elif recent_return > threshold:  # Positive return
            reward += 25
        if enhanced_s[-5] < 0.5:  # Weak trend signal
            reward -= 30  # Penalize holding in weak trend

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]