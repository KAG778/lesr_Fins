import numpy as np
import pandas as pd

def calculate_sma(prices, window):
    return pd.Series(prices).rolling(window=window).mean().values[-1]

def calculate_ema(prices, window):
    return pd.Series(prices).ewm(span=window, adjust=False).mean().values[-1]

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean() if len(deltas) > 0 else 0
    loss = -np.where(deltas < 0, deltas, 0).mean() if len(deltas) > 0 else 0
    rs = gain / loss if loss != 0 else np.inf
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
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

    features = []

    # Trend Indicators
    features.append(calculate_sma(closing_prices, 5))
    features.append(calculate_sma(closing_prices, 10))
    features.append(calculate_sma(closing_prices, 20))
    features.append(calculate_ema(closing_prices, 5))
    features.append(calculate_ema(closing_prices, 10))
    features.append(calculate_ema(closing_prices, 20))

    # Momentum Indicators
    features.append(calculate_rsi(closing_prices, 5))
    features.append(calculate_rsi(closing_prices, 14))
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) > 1 else 0
    features.append(momentum)

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices)) if len(closing_prices) >= 5 else np.nan
    historical_volatility_20 = np.std(np.diff(closing_prices)) if len(closing_prices) >= 20 else np.nan
    features.append(historical_volatility_5)
    features.append(historical_volatility_20)
    features.append(calculate_atr(high_prices, lows, closing_prices, 14))

    # Volume-Price Relationship
    obv = np.zeros(len(closing_prices))
    for i in range(1, len(closing_prices)):
        if closing_prices[i] > closing_prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closing_prices[i] < closing_prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    features.append(obv[-1])
    features.append(volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 0)  # Volume ratio

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else 0
    features.append(volatility_ratio)
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] if len(closing_prices) > 1 else 0
    features.append(trend_strength)
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    features.append(price_position)
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0
    features.append(volume_ratio_regime)

    # Threshold Feature
    features.append(2 * historical_volatility_5 if historical_volatility_5 is not None else 0)  # 2x historical volatility for adaptive threshold

    # Combine original state with new features
    enhanced_s = np.concatenate((s, features))
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0
    historical_volatility = np.std(np.diff(closing_prices)) if len(closing_prices) >= 5 else 1  # Avoid division by zero
    threshold = 2 * historical_volatility  # Volatility-adaptive threshold

    reward = 0

    if position_flag == 0.0:  # Not holding
        if recent_return > threshold:  # Strong Buy signal
            reward += 50
        elif recent_return < -threshold:  # Strong Sell signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Weakening trend
            reward -= 50
        else:
            reward += 20  # Reward for holding during a positive move

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]