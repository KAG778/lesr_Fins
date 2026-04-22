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
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema[-1]

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12 - ema_26
    signal_line = calculate_ema(prices, 9)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_volatility(returns, window):
    if len(returns) < window or np.std(returns) == 0:
        return np.nan
    return np.std(returns[-window:])

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    enhanced_features = []

    # Trend Indicators
    for window in [5, 10, 20, 50]:
        enhanced_features.append(calculate_sma(closing_prices, window))  # SMA
        enhanced_features.append(calculate_ema(closing_prices, window))  # EMA
        enhanced_features.append(closing_prices[-1] / calculate_sma(closing_prices, window))  # Price/SMA ratio

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    macd, signal, histogram = calculate_macd(closing_prices)
    
    enhanced_features.extend([rsi_5, rsi_10, macd, signal, histogram])

    # Volatility Indicators
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_vol_5 = calculate_volatility(daily_returns, 5)
    historical_vol_20 = calculate_volatility(daily_returns, 20)
    atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(high_prices[:-1] - closing_prices[:-1], 
                                         closing_prices[:-1] - low_prices[:-1])))  # ATR
    volatility_ratio = historical_vol_5 / historical_vol_20 if historical_vol_20 else np.nan
    
    enhanced_features.extend([historical_vol_5, historical_vol_20, atr, volatility_ratio])

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], 
                             np.where(np.diff(closing_prices) < 0, -volumes[1:], 0)))
    volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) != 0 else np.nan
    
    enhanced_features.extend([obv[-1], volume_ratio])
    
    # Market Regime Detection
    trend_strength = np.polyfit(np.arange(len(closing_prices)), closing_prices, 1)[0]  # Linear regression slope
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    volume_ratio_regime = (np.mean(volumes[-5:]) / np.mean(volumes[-20:])) > 2.0 if np.mean(volumes[-20:]) != 0 else np.nan
    
    enhanced_features.extend([trend_strength, price_position, volume_ratio_regime])
    
    # Combine original state with enhanced features
    enhanced_s = np.concatenate((s, np.array(enhanced_features)))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Position flag (1.0=holding, 0.0=not holding)
    closing_prices = enhanced_s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    if len(daily_returns) < 1:
        return 0  # Insufficient data for reward calculation

    recent_return = daily_returns[-1] * 100  # Convert to percentage
    historical_vol = np.std(daily_returns) * 100  # Convert to percentage
    threshold = 2 * historical_vol if historical_vol != 0 else 0  # Volatility-adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if enhanced_s[-5] > 70 and recent_return > threshold:  # RSI example condition for overbought
            reward += 50  # Strong buy signal
        elif recent_return < -threshold:
            reward -= 30  # Penalize for negative return
        else:
            reward += 10  # Neutral reward

    elif position_flag == 1:  # Holding
        if recent_return < -threshold:  # Penalizing strong down moves
            reward -= 50
        elif enhanced_s[-5] < 30 and recent_return > 0:  # RSI example condition for oversold
            reward += 60  # Positive reward for holding in an uptrend
        else:
            reward += 20  # Neutral reward for holding

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]