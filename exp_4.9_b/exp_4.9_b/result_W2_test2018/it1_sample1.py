import numpy as np
import pandas as pd

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window + 1:]:
        ema = (price * alpha) + (ema * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    if np.isnan(ema_12) or np.isnan(ema_26):
        return np.nan, np.nan, np.nan
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(prices[-26:], 9)
    histogram = macd_line - signal_line if not np.isnan(signal_line) else np.nan
    return macd_line, signal_line, histogram

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * 100  # Convert to percentage

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = pd.Series(tr).rolling(window=window).mean().values
    return atr[-1] if len(atr) > 0 else np.nan

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    enhanced_features = []

    # Multi-timeframe Trend Indicators
    enhanced_features.append(calculate_sma(closing_prices, 5))
    enhanced_features.append(calculate_sma(closing_prices, 10))
    enhanced_features.append(calculate_sma(closing_prices, 20))
    enhanced_features.append(calculate_ema(closing_prices, 5))
    enhanced_features.append(calculate_ema(closing_prices, 10))
    enhanced_features.append(calculate_ema(closing_prices, 20))
    
    # Momentum Indicators
    enhanced_features.append(calculate_rsi(closing_prices, 5))
    enhanced_features.append(calculate_rsi(closing_prices, 10))
    enhanced_features.append(calculate_rsi(closing_prices, 14))
    macd_line, signal_line, histogram = calculate_macd(closing_prices)
    enhanced_features.append(macd_line)
    enhanced_features.append(signal_line)
    enhanced_features.append(histogram)

    # Volatility Indicators
    enhanced_features.append(calculate_volatility(closing_prices, 5))
    enhanced_features.append(calculate_volatility(closing_prices, 20))
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    enhanced_features.append(atr)

    # Volume-Price Relationship
    obv = np.zeros(len(volumes))
    for i in range(1, len(volumes)):
        if closing_prices[i] > closing_prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif closing_prices[i] < closing_prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    enhanced_features.append(obv[-1])  # Latest OBV

    # Market Regime Detection
    volatility_5 = calculate_volatility(closing_prices, 5)
    volatility_20 = calculate_volatility(closing_prices, 20)
    enhanced_features.append(volatility_5 / volatility_20 if volatility_20 != 0 else np.nan)
    
    trend_strength = np.corrcoef(np.arange(len(closing_prices)), closing_prices)[0, 1]  # Linear trend
    enhanced_features.append(trend_strength)
    
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    enhanced_features.append(price_position)

    volume_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan
    enhanced_features.append(volume_ratio)

    # Additional Features
    # Adding exponential moving average of volume
    ewm_volume = pd.Series(volumes).ewm(span=20).mean().values
    enhanced_features.append(ewm_volume[-1])

    # Adding volatility expansion feature
    volatility_expansion = (volatility_5 - volatility_20) / (volatility_20 + 1e-10)
    enhanced_features.append(volatility_expansion)

    # Adding relative strength index divergence
    rsi_divergence = np.mean(np.diff(np.array([calculate_rsi(closing_prices, 5), calculate_rsi(closing_prices, 10)])))
    enhanced_features.append(rsi_divergence)

    # Combine original state with enhanced features
    enhanced_s = np.concatenate((s, enhanced_features))
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag (1.0 = holding, 0.0 = not holding)
    closing_prices = enhanced_s[0:20]  # Recent closing prices
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    
    # Calculate historical volatility from closing prices
    historical_vol = calculate_volatility(closing_prices, 20)
    threshold = 2 * historical_vol if historical_vol > 0 else 0  # Adaptive threshold based on historical volatility
    
    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Large downtrend
            reward -= 50
    else:  # Holding
        if recent_return > 0:  # Continuing uptrend
            reward += 10
        elif recent_return < -threshold:  # Significant drop while holding
            reward -= 100
            
    return np.clip(reward, -100, 100)