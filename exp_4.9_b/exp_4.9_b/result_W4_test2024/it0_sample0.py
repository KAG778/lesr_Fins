import numpy as np
import pandas as pd

def calculate_sma(prices, window):
    return pd.Series(prices).rolling(window=window).mean().values

def calculate_ema(prices, window):
    return pd.Series(prices).ewm(span=window, adjust=False).mean().values

def calculate_rsi(prices, window):
    delta = np.diff(prices)
    gain = (delta[delta > 0]).sum() / window
    loss = (-delta[delta < 0]).sum() / window
    rs = gain / loss if loss != 0 else np.inf
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line[-1], signal_line[-1], histogram[-1]

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    return pd.Series(tr).rolling(window=window).mean().values[-1]

def revise_state(raw_state):
    closing_prices = raw_state[0:20]
    opening_prices = raw_state[20:40]
    high_prices = raw_state[40:60]
    low_prices = raw_state[60:80]
    volumes = raw_state[80:100]
    adjusted_closing_prices = raw_state[100:120]

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1]
    sma_10 = calculate_sma(closing_prices, 10)[-1]
    sma_20 = calculate_sma(closing_prices, 20)[-1]
    price_vs_sma_5 = closing_prices[-1] / sma_5
    price_vs_sma_10 = closing_prices[-1] / sma_10
    price_vs_sma_20 = closing_prices[-1] / sma_20

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    macd_line, signal_line, macd_histogram = calculate_macd(closing_prices)

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) * 100
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 0

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 > 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0

    # Construct the enhanced state
    enhanced_s = np.concatenate([
        raw_state,  # Original state
        np.array([sma_5, sma_10, sma_20, price_vs_sma_5, price_vs_sma_10, price_vs_sma_20,
                  rsi_5, rsi_10, rsi_14, macd_line, signal_line, macd_histogram,
                  historical_volatility_5, historical_volatility_20, atr,
                  obv[-1], volume_ratio,
                  volatility_ratio, trend_strength, price_position, volume_ratio_regime])
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100

    threshold = 2 * historical_volatility

    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold and enhanced_s[120] > 0:  # Strong uptrend, oversold
            reward += 50
        else:
            reward -= 10  # Penalize uncertain conditions
    else:  # Holding
        if recent_return < -threshold:
            reward -= 50  # Penalize if the price drops significantly
        elif enhanced_s[120] < 0:  # Trend weakens
            reward -= 20
        else:
            reward += 10  # Reward for holding during uptrend

    return np.clip(reward, -100, 100)  # Clip the reward to the range [-100, 100]