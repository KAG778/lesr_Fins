import numpy as np
import pandas as pd

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append((price - ema[-1]) * alpha + ema[-1])
    return np.array(ema)

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else np.nan

def calculate_volatility(prices, window):
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) if len(returns) >= window else np.nan

def calculate_obv(prices, volumes):
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        obv[i] = obv[i - 1] + volumes[i] if prices[i] > prices[i - 1] else obv[i - 1] - volumes[i]
    return obv[-1]

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    features = []

    # Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else np.nan
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    ema_10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5) if len(closing_prices) >= 5 else np.nan
    rsi_14 = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) >= 2 else np.nan

    # Volatility Indicators
    historical_volatility_5 = calculate_volatility(closing_prices, 5)
    historical_volatility_20 = calculate_volatility(closing_prices, 20)
    atr_14 = calculate_atr(high_prices, lows, closing_prices, 14)

    # Volume-Price Relationship
    obv = calculate_obv(closing_prices, volumes)
    volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else np.nan

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2 if len(closing_prices) >= 2 else np.nan
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Construct the enhanced state
    enhanced_s = np.concatenate([
        s,
        np.array([
            sma_5, sma_10, sma_20, ema_5, ema_10,
            rsi_5, rsi_14, momentum,
            historical_volatility_5, historical_volatility_20, atr_14,
            obv, volume_ratio,
            volatility_ratio, trend_strength, price_position, volume_ratio_regime,
            # Additional features
            (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2],  # Daily return
            (closing_prices[-1] - sma_20) / sma_20,  # Price relative to 20-day SMA
            (sma_5 - sma_20) / sma_20,  # SMA spread
            (closing_prices[-1] - np.mean(closing_prices[-5:])) / np.std(closing_prices[-5:]),  # Z-score of the last price
            (closing_prices[-1] - np.mean(closing_prices[-10:])) / np.std(closing_prices[-10:]),  # Z-score of the last price over 10 days
            np.max(closing_prices) - np.min(closing_prices),  # Range of closing prices
            np.mean(volumes[-10:]),  # Average volume over the last 10 days
            np.mean(volumes[-5:]) / np.mean(volumes[-20:]),  # 5-day average volume / 20-day average volume
        ])
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return and historical volatility
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) >= 2 else 0
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 1 else 1  # Avoid division by zero

    # Use 2x historical volatility as threshold for rewards
    threshold = 2 * historical_vol

    reward = 0
    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong Buy signal
            reward += 50
        elif recent_return < -threshold:  # Strong Sell signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Penalize for holding in a downturn
            reward -= 50
        else:
            reward += 20  # Reward for holding during a positive move

    return np.clip(reward, -100, 100)  # Ensure reward is within specified range