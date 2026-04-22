import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

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
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else np.nan

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    returns = np.diff(prices) / prices[:-1] * 100
    return np.std(returns[-window:]) if len(returns) >= window else np.nan

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    enhanced_features = []

    # Trend Indicators
    enhanced_features.append(calculate_sma(closing_prices, 5))  # 5-day SMA
    enhanced_features.append(calculate_sma(closing_prices, 10))  # 10-day SMA
    enhanced_features.append(calculate_sma(closing_prices, 20))  # 20-day SMA
    enhanced_features.append(calculate_ema(closing_prices, 5))  # 5-day EMA
    enhanced_features.append(calculate_ema(closing_prices, 10))  # 10-day EMA
    enhanced_features.append(calculate_ema(closing_prices, 20))  # 20-day EMA

    # Momentum Indicators
    enhanced_features.append(calculate_rsi(closing_prices, 5))  # 5-day RSI
    enhanced_features.append(calculate_rsi(closing_prices, 14))  # 14-day RSI
    enhanced_features.append(calculate_rsi(closing_prices, 21))  # 21-day RSI

    # Additional Momentum Indicators
    enhanced_features.append(closing_prices[-1] - closing_prices[-2])  # Daily price change (momentum)
    
    # Volatility Indicators
    enhanced_features.append(calculate_volatility(closing_prices, 5))  # 5-day historical volatility
    enhanced_features.append(calculate_volatility(closing_prices, 20))  # 20-day historical volatility
    enhanced_features.append(calculate_atr(high_prices, lows, closing_prices, 14))  # ATR

    # Volume Indicators
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], np.where(np.diff(closing_prices) < 0, -volumes[1:], 0)))
    enhanced_features.append(obv)  # On-Balance Volume
    enhanced_features.append(np.mean(volumes[-5:]) / np.mean(volumes[-20:]))  # Volume ratio

    # Market Regime Features
    volatility_ratio = enhanced_features[-2] / enhanced_features[-1] if enhanced_features[-1] > 0 else np.nan
    enhanced_features.append(volatility_ratio)  # Volatility ratio
    enhanced_features.append((closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])))  # Price position in range
    trend_strength = np.polyfit(range(len(closing_prices)), closing_prices, 1)[0]  # Linear regression slope as trend strength
    enhanced_features.append(trend_strength)  # Trend strength

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(enhanced_features)))
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    position_flag = enhanced_s[-1]
    
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    threshold = 2 * historical_vol  # Adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if enhanced_s[-4] > 1:  # If trend strength is strong
            reward += 50  # Strong buy signal
        if recent_return > threshold:  # Recent return exceeds threshold
            reward += 30
        elif recent_return < -threshold:  # Recent return below threshold
            reward -= 20

    elif position_flag == 1:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        if recent_return > threshold:  # Positive return
            reward += 25
        if enhanced_s[-4] < 0.5:  # Weak trend
            reward -= 30  # Consider selling

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]