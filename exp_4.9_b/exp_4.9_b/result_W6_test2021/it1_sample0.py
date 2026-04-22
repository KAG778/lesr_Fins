import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window-1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema[-1]

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.maximum(np.abs(highs[-window:] - closes[-window:]), 
                               np.abs(lows[-window:] - closes[-window:])))
    return np.mean(tr)

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    if np.isnan(ema_12) or np.isnan(ema_26):
        return np.nan, np.nan, np.nan
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(prices[-26:], 9)  # Signal line is EMA of MACD line
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_obv(volumes, closing_prices):
    obv = np.zeros_like(closing_prices)
    for i in range(1, len(closing_prices)):
        if closing_prices[i] > closing_prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closing_prices[i] < closing_prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv[-1]

def calculate_trend_strength(prices):
    from sklearn.linear_model import LinearRegression
    x = np.arange(len(prices)).reshape(-1, 1)  # Days as a feature
    y = prices.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    r_squared = model.score(x, y)
    return r_squared

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    features = []
    
    # Trend Indicators
    features.append(calculate_sma(closing_prices, 5))
    features.append(calculate_sma(closing_prices, 10))
    features.append(calculate_sma(closing_prices, 20))
    features.append(calculate_ema(closing_prices, 5))
    features.append(calculate_ema(closing_prices, 10))
    
    # Momentum Indicators
    features.append(calculate_rsi(closing_prices, 14))
    macd_line, signal_line, histogram = calculate_macd(closing_prices)
    features.append(macd_line)
    features.append(signal_line)
    features.append(histogram)

    # Volatility Indicators
    features.append(calculate_atr(high_prices, low_prices, closing_prices, 14))
    historical_volatility_5 = np.std(np.diff(closing_prices)) * np.sqrt(252)  # Annualized
    features.append(historical_volatility_5)
    features.append(np.std(np.diff(closing_prices[-20:])) * np.sqrt(252))  # 20-day annualized volatility

    # Volume Indicators
    obv = calculate_obv(volumes, closing_prices)
    features.append(obv)
    features.append(volumes[-1] / np.mean(volumes[-20:]))  # Volume ratio

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / np.std(np.diff(closing_prices[-20:])) if np.std(np.diff(closing_prices[-20:])) != 0 else np.nan
    features.append(volatility_ratio)
    trend_strength = calculate_trend_strength(closing_prices)
    features.append(trend_strength)
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    features.append(price_position)
    volume_ratio_regime = (np.mean(volumes[-5:]) / np.mean(volumes[-20:])) if np.mean(volumes[-20:]) != 0 else np.nan
    features.append(volume_ratio_regime)

    # Adding new features based on combinations and ratios
    features.append(np.max(closing_prices[-5:]) / np.mean(closing_prices[-20:]))  # Max over last 5 days vs Mean
    features.append(np.mean(closing_prices[-5:]) / np.mean(closing_prices[-20:]))  # Mean over last 5 days vs Mean
    features.append(np.mean(np.diff(closing_prices[-5:])) * 100)  # Average daily return over last 5 days
    features.append(np.mean(np.diff(closing_prices[-10:])) * 100)  # Average daily return over last 10 days
    features.append(np.std(np.diff(closing_prices[-5:])) * 100)  # Volatility over last 5 days
    features.append(np.std(np.diff(closing_prices[-10:])) * 100)  # Volatility over last 10 days

    # Combine original state with enhanced features
    enhanced_s = np.concatenate((s, np.array(features)))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100  # Recent returns in percentage
    
    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Volatility adaptive threshold
    
    recent_return = returns[-1] if len(returns) > 0 else 0  # Latest return in percentage
    reward = 0
    
    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50
    elif position_flag == 1:  # Holding
        if recent_return > 0:  # Positive return
            reward += 20
        elif recent_return < -threshold:  # Significant drop
            reward -= 50
    
    # Penalize uncertain market conditions
    if np.isnan(recent_return) or np.std(returns) < (0.5 * historical_vol):
        reward -= 20  # Choppy market

    return np.clip(reward, -100, 100)  # Ensure reward is within limits