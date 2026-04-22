import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window), 'valid') / window

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
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else np.nan

def calculate_obv(prices, volumes):
    obv = np.zeros_like(prices)
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif prices[i] < prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv

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
    enhanced_features.append(closing_prices[-1] - calculate_sma(closing_prices, 10))  # Price above 10-day SMA
    enhanced_features.append(closing_prices[-1] - calculate_sma(closing_prices, 20))  # Price above 20-day SMA

    # Momentum Indicators
    enhanced_features.append(calculate_rsi(closing_prices, 5))  # 5-day RSI
    enhanced_features.append(calculate_rsi(closing_prices, 14))  # 14-day RSI
    enhanced_features.append(closing_prices[-1] / closing_prices[-2] - 1)  # Daily return

    # MACD
    ema_12 = calculate_ema(closing_prices, 12)
    ema_26 = calculate_ema(closing_prices, 26)
    if not np.isnan(ema_12) and not np.isnan(ema_26):
        macd_line = ema_12 - ema_26
        signal_line = calculate_ema([macd_line] + [0] * (len(closing_prices) - 1), 9)  # Signal line
        enhanced_features.append(macd_line)
        enhanced_features.append(macd_line - signal_line)  # MACD Histogram

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 5 else np.nan
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else np.nan
    atr = calculate_atr(high_prices, lows, closing_prices, 14)
    
    enhanced_features.append(historical_volatility_5)  # 5-day historical volatility
    enhanced_features.append(historical_volatility_20)  # 20-day historical volatility
    enhanced_features.append(atr)  # ATR

    # Volume Indicators
    obv = calculate_obv(closing_prices, volumes)
    enhanced_features.append(obv)  # OBV
    volume_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan
    enhanced_features.append(volume_ratio)  # Volume Ratio

    # Market Regime Detection
    volatility_ratio = (historical_volatility_5 / historical_volatility_20) if historical_volatility_20 != 0 else np.nan
    trend_strength = np.corrcoef(np.arange(len(closing_prices)), closing_prices)[0, 1]  # Linear trend
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if len(closing_prices) >= 20 else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    enhanced_features.append(volatility_ratio)  # Volatility ratio
    enhanced_features.append(trend_strength)  # Trend strength
    enhanced_features.append(price_position)  # Price position
    enhanced_features.append(volume_ratio_regime)  # Volume ratio regime

    # Combine original state and new features
    enhanced_s = np.concatenate((s, np.array(enhanced_features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    position_flag = enhanced_s[-1]

    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0

    # Calculate historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(daily_returns) if len(daily_returns) > 0 else 0
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong buy signal
            reward += 50
        elif recent_return < -threshold:  # Strong sell signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        elif recent_return > threshold:  # Positive return
            reward += 25

    # Penalize for weak trend
    trend_strength = enhanced_s[-4]  # Assuming trend_strength is at index -4
    if trend_strength < 0.5:  # Weak trend
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]