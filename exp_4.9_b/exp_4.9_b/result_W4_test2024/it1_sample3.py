import numpy as np
import pandas as pd

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = [prices[0]]  # Start with the first price
    for price in prices[1:]:
        ema.append((price - ema[-1]) * alpha + ema[-1])
    return np.array(ema)

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-window:]) if len(tr) >= window else np.nan
    return atr

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

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else np.nan
    ema_5 = calculate_ema(closing_prices, 5)[-1]
    ema_10 = calculate_ema(closing_prices, 10)[-1]
    trend_difference_5_20 = sma_5 - sma_20 if not np.isnan(sma_5) and not np.isnan(sma_20) else np.nan
    price_above_sma_20 = (closing_prices[-1] - sma_20) / sma_20 if sma_20 else np.nan

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5) if len(closing_prices) >= 5 else np.nan
    rsi_14 = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 else np.nan

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices)) if len(closing_prices) >= 5 else np.nan
    historical_volatility_20 = np.std(np.diff(closing_prices)) if len(closing_prices) >= 20 else np.nan
    atr = calculate_atr(high_prices, lows, closing_prices, 14)  # ATR over 14 days

    # Volume-Price Relationship
    obv = calculate_obv(closing_prices, volumes)
    volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else np.nan

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 > 0 else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] if len(closing_prices) > 1 else np.nan
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Compile all features into an enhanced state
    enhanced_s = np.concatenate((
        s,
        np.array([sma_5, sma_10, sma_20, ema_5, ema_10, trend_difference_5_20, price_above_sma_20,
                  rsi_5, rsi_14, momentum,
                  historical_volatility_5, historical_volatility_20, atr,
                  obv, volume_ratio,
                  volatility_ratio, trend_strength, price_position, volume_ratio_regime])
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Get the position flag
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) >= 2 else 0
    historical_vol = np.std(np.diff(closing_prices)) if len(closing_prices) >= 5 else 1  # Avoid division by zero

    # Use adaptive thresholds
    threshold = 2 * historical_vol  # Volatility-adaptive threshold
    reward = 0

    if position_flag == 0.0:  # Not holding
        if recent_return > threshold:  # Strong uptrend signal
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Weakening trend
            reward -= 50
        else:
            reward += 20  # Reward for holding during a positive move

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range