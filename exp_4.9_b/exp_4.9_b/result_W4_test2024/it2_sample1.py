import numpy as np
import pandas as pd

def calculate_sma(prices, window):
    return pd.Series(prices).rolling(window=window).mean().values[-1] if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema[-1]

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else np.nan

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns)

def calculate_obv(prices, volumes):
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif prices[i] < prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv[-1]

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    features = []

    # Multi-timeframe Trend Indicators
    features.append(calculate_sma(closing_prices, 5))
    features.append(calculate_sma(closing_prices, 10))
    features.append(calculate_sma(closing_prices, 20))
    features.append(calculate_ema(closing_prices, 5))
    features.append(calculate_ema(closing_prices, 10))
    features.append(calculate_ema(closing_prices, 20))
    features.append(closing_prices[-1] - calculate_sma(closing_prices, 20))  # Price relative to 20-day SMA
    features.append(closing_prices[-1] > calculate_sma(closing_prices, 20))  # Boolean: above 20-day SMA

    # Momentum Indicators
    features.append(calculate_rsi(closing_prices, 5))
    features.append(calculate_rsi(closing_prices, 14))
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0
    features.append(momentum)  # Momentum rate of change

    # Volatility Indicators
    features.append(calculate_volatility(closing_prices, 5))
    features.append(calculate_volatility(closing_prices, 20))
    features.append(calculate_atr(high_prices, low_prices, closing_prices, 14))  # ATR

    # Volume-Price Relationships
    features.append(calculate_obv(closing_prices, volumes))
    features.append(volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else np.nan)  # Volume ratio

    # Market Regime Detection
    volatility_ratio = features[10] / features[11] if features[11] != 0 else np.nan  # 5-day / 20-day volatility
    features.append(volatility_ratio)
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]  # Linear regression R²
    features.append(trend_strength)
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    features.append(price_position)
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan  # 5-day / 20-day average volume
    features.append(volume_ratio_regime)

    # New features for enhanced state representation
    features.append(np.std(closing_prices))  # Standard deviation of closing prices
    features.append(np.mean(closing_prices))  # Mean of closing prices
    features.append(np.median(closing_prices))  # Median of closing prices
    features.append(np.max(closing_prices) - np.min(closing_prices))  # Range of closing prices
    features.append(np.mean(volumes[-5:]))  # Last 5 days average volume

    # Combine original state with new features
    enhanced_s = np.concatenate((s, features))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Get the position flag
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0
    historical_volatility = np.std(np.diff(closing_prices)) if len(closing_prices) >= 2 else 1  # Avoid division by zero

    # Use relative thresholds based on historical volatility
    threshold = 2 * historical_volatility  # Adaptive threshold

    reward = 0

    if position_flag == 0.0:  # Not holding
        if recent_return > threshold:  # Strong Buy signal
            reward += 50
        elif recent_return < -threshold:  # Strong Sell signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Weakening trend
            reward -= 50
        elif recent_return > 0:  # Positive return
            reward += 10  # Reward for holding during a positive move

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range