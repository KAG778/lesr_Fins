import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema[-1]

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[1:] - lows[1:], 
                   np.maximum(np.abs(highs[1:] - closes[:-1]), 
                              np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    features = []

    # Trend Indicators
    for window in [5, 10, 20, 50]:
        features.append(calculate_sma(closing_prices, window))
        features.append(calculate_ema(closing_prices, window))
        features.append(closing_prices[-1] - calculate_sma(closing_prices, window))  # Price vs SMA
        features.append(closing_prices[-1] - calculate_ema(closing_prices, window))  # Price vs EMA
    
    # Momentum Indicators
    features.append(calculate_rsi(closing_prices, 5))
    features.append(calculate_rsi(closing_prices, 14))
    
    # Volatility Indicators
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    features.append(np.std(daily_returns) * 100)  # Historical volatility
    features.append(calculate_atr(high_prices, lows, closing_prices, 14))  # ATR
    features.append(np.std(daily_returns[-5:]) * 100)  # Short-term volatility
    features.append(np.std(daily_returns[-20:]) * 100)  # Medium-term volatility

    # Volume Indicators
    obv = np.cumsum(np.where(np.diff(adjusted_closing_prices) > 0, volumes[1:], 
                             np.where(np.diff(adjusted_closing_prices) < 0, -volumes[1:], 0)))
    features.append(obv[-1])  # On-Balance Volume
    features.append(volumes[-1] / np.mean(volumes))  # Volume ratio
    features.append(volumes[-5:].mean() / volumes[-20:].mean())  # Short vs long-term volume

    # Market Regime Detection
    volatility_ratio = features[6] / features[7] if features[7] != 0 else 0  # Historical volatility / ATR
    trend_strength = np.polyfit(range(len(closing_prices)), closing_prices, 1)[0]  # Linear regression slope
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    features.append(volatility_ratio)  # Volatility ratio
    features.append(trend_strength)  # Trend strength
    features.append(price_position)  # Price position in range

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(features)))
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    if len(daily_returns) < 1:
        return 0  # Insufficient data

    recent_return = daily_returns[-1] * 100  # Convert to percentage
    historical_vol = np.std(daily_returns) * 100  # Convert to percentage
    threshold = 2 * historical_vol if historical_vol != 0 else 0  # Volatility-adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if enhanced_s[-5] < 30 and recent_return > threshold:  # Buy signal condition
            reward += 50  # Strong buy signal
        elif recent_return < -threshold:  # If recent return is significantly negative
            reward -= 30  # Penalty for potential missed opportunity
        else:
            reward += 10  # Neutral reward
    else:  # Holding
        if recent_return < -threshold:  # If recent return is significantly negative
            reward -= 50  # Penalty for holding during downturn
        elif enhanced_s[-5] > 70 and recent_return > 0:  # Sell signal condition
            reward += 60  # Positive reward for holding
        else:
            reward += 20  # Neutral reward for holding

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]