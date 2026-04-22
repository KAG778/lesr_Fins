import numpy as np
import pandas as pd

def calculate_sma(prices, period):
    return pd.Series(prices).rolling(window=period).mean().to_numpy()

def calculate_ema(prices, period):
    return pd.Series(prices).ewm(span=period, adjust=False).mean().to_numpy()

def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(window=period).mean().to_numpy()
    rs = avg_gain / (avg_loss + 1e-10)  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = pd.Series(tr).rolling(window=period).mean().to_numpy()
    return atr

def revise_state(raw_state):
    closing_prices = raw_state[0:20]
    opening_prices = raw_state[20:39]
    high_prices = raw_state[40:59]
    low_prices = raw_state[60:79]
    volumes = raw_state[80:99]
    
    # Calculate features
    features = []

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1]
    sma_10 = calculate_sma(closing_prices, 10)[-1]
    sma_20 = calculate_sma(closing_prices, 20)[-1]
    ema_5 = calculate_ema(closing_prices, 5)[-1]
    ema_10 = calculate_ema(closing_prices, 10)[-1]
    
    features.extend([sma_5, sma_10, sma_20, ema_5, ema_10])
    features.append((closing_prices[-1] - sma_10) / sma_10)  # Price relative to 10-day SMA

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)[-1]
    rsi_10 = calculate_rsi(closing_prices, 10)[-1]
    rsi_14 = calculate_rsi(closing_prices, 14)[-1]
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]  # Rate of change

    features.extend([rsi_5, rsi_10, rsi_14, momentum])

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1])  # 5-day volatility
    historical_volatility_20 = np.std(np.diff(closing_prices) / closing_prices[:-1])  # 20-day volatility
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)[-1]

    features.extend([historical_volatility_5, historical_volatility_20, atr])

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], 
                              np.where(np.diff(closing_prices) < 0, -volumes[1:], 0)))
    volume_avg_5 = np.mean(volumes[-5:])
    volume_avg_20 = np.mean(volumes[-20:])
    volume_ratio = volume_avg_5 / volume_avg_20

    features.extend([obv[-1], volume_ratio])

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / (historical_volatility_20 + 1e-10)  # Prevent division by zero
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]  # R² trend strength
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices) + 1e-10)  # Normalized position
    volume_ratio_regime = volume_avg_5 / volume_avg_20  # Volume ratio regime

    features.extend([volatility_ratio, trend_strength, price_position, volume_ratio_regime])

    enhanced_state = np.concatenate([raw_state, features])
    return enhanced_state

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100  # Historical volatility

    threshold = 2 * historical_vol  # Adaptive threshold based on historical volatility
    reward = 0

    # Reward logic based on position
    if position_flag == 0.0:  # Not holding
        if recent_return > threshold:  # Clear BUY signal
            reward += 50
        elif recent_return < -threshold:  # Negative signal
            reward -= 20
    else:  # Holding
        if recent_return < -threshold:  # Consider selling if trend weakens
            reward -= 30
        elif recent_return > 0:  # Positive return, HOLD
            reward += 20
        else:  # Uncertain condition
            reward -= 10

    return np.clip(reward, -100, 100)  # Clip reward to range [-100, 100]