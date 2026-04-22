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

def calculate_volatility_ratio(closing_prices):
    return np.std(np.diff(closing_prices)) / np.std(np.diff(closing_prices[-20:])) if np.std(np.diff(closing_prices[-20:])) != 0 else np.nan

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Feature Calculation
    features = []

    # A. Multi-timeframe Trend Indicators
    features.append(calculate_sma(closing_prices, 5))
    features.append(calculate_sma(closing_prices, 10))
    features.append(calculate_sma(closing_prices, 20))
    features.append(calculate_ema(closing_prices, 5))
    features.append(calculate_ema(closing_prices, 10))
    
    # B. Momentum Indicators
    features.append(calculate_rsi(closing_prices, 5))
    features.append(calculate_rsi(closing_prices, 14))
    
    # C. Volatility Indicators
    features.append(calculate_atr(high_prices, low_prices, 14))  # ATR
    features.append(np.std(np.diff(closing_prices)) * np.sqrt(252))  # 5-day historical volatility
    features.append(calculate_volatility_ratio(closing_prices))  # Volatility ratio
    
    # D. Volume-Price Relationship
    obv = calculate_obv(volumes, closing_prices)
    features.append(obv)  # Most recent OBV
    features.append(volumes[-1] / np.mean(volumes[-20:]))  # Volume ratio
    
    # E. Market Regime Detection
    trend_strength = calculate_trend_strength(closing_prices)
    features.append(trend_strength)  # R² of closing prices
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    features.append(price_position)  # Price position within range
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan
    features.append(volume_ratio_regime)  # Volume ratio for regime detection

    # Adding features to the state
    enhanced_s = np.concatenate((s, np.array(features)))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Position flag (1.0 for holding, 0.0 for not holding)
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Recent return in percentage
    
    # Calculate historical volatility
    historical_vol = np.std(np.diff(closing_prices)) * 100  # Convert to percentage
    threshold = 2 * historical_vol  # Volatility adaptive threshold

    # Reward calculation
    reward = 0
    
    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50
    elif position_flag == 1:  # Holding
        if recent_return > 0:  # Positive return
            reward += 10
        elif recent_return < -threshold:  # Significant drop
            reward -= 50

    # Penalize uncertain/choppy market conditions
    volatility_ratio = enhanced_s[-4]  # Retrieve volatility ratio
    if volatility_ratio < 1:  # Low volatility
        reward -= 20
    
    return np.clip(reward, -100, 100)  # Ensure reward is within limits