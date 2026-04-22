import numpy as np
import pandas as pd

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

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
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * 100

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
    features.append((closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100)  # Daily return

    # Momentum Indicators
    features.append(calculate_rsi(closing_prices, 5))
    features.append(calculate_rsi(closing_prices, 10))
    features.append(calculate_rsi(closing_prices, 14))

    # Volatility Indicators
    features.append(calculate_volatility(closing_prices, 5))
    features.append(calculate_volatility(closing_prices, 20))
    features.append(calculate_atr(high_prices, lows, closing_prices, 14))

    # Volume-Price Relationship
    features.append(calculate_obv(closing_prices, volumes))
    features.append(volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else np.nan)  # Volume ratio

    # Market Regime Detection
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    features.append(historical_volatility)  # Historical volatility
    features.append(features[5] / features[6] if features[6] > 0 else np.nan)  # Volatility ratio
    features.append(np.corrcoef(closing_prices, volumes)[0, 1])  # Volume-price correlation
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2 if len(closing_prices) > 1 else np.nan
    features.append(trend_strength)  # Trend strength
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    features.append(price_position)  # Price position

    # Compile all features into an enhanced state
    enhanced_s = np.concatenate((s, features))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Get the position flag
    closing_prices = enhanced_s[0:20]
    
    # Calculate daily return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0

    # Calculate historical volatility
    historical_vol = np.std(np.diff(closing_prices)) * 100 if len(closing_prices) >= 5 else 1  # Avoid division by zero

    # Use relative threshold
    threshold = 2 * historical_vol  # Adaptive threshold

    reward = 0

    if position_flag == 0.0:  # Not holding
        if recent_return > threshold and enhanced_s[120] > 0:  # Assuming 120th index is trend strength
            reward += 50  # Strong Buy signal
        elif recent_return < -threshold:  # Strong Sell signal
            reward -= 50  # Strong Sell signal
    else:  # Holding
        if recent_return < -threshold:
            reward -= 50  # Penalize for holding in a downturn
        elif recent_return > 0:  # Positive return
            reward += 20  # Reward for holding during a positive move
        elif enhanced_s[120] < 0.1:  # Assuming trend strength is low
            reward -= 10  # Consider selling in a weak trend

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]