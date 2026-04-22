import numpy as np
import pandas as pd

def calculate_sma(prices, window):
    return pd.Series(prices).rolling(window=window).mean().values

def calculate_ema(prices, window):
    return pd.Series(prices).ewm(span=window, adjust=False).mean().values

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean().values
    avg_loss = pd.Series(loss).rolling(window=window).mean().values
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = pd.Series(tr).rolling(window=window).mean().values
    atr = np.concatenate(([np.nan], atr))  # Align lengths
    return atr

def revise_state(raw_state):
    closing_prices = raw_state[0:20]
    opening_prices = raw_state[20:40]
    high_prices = raw_state[40:60]
    low_prices = raw_state[60:80]
    volumes = raw_state[80:100]
    adjusted_closing_prices = raw_state[100:120]

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    macd_line, signal_line, macd_hist = calculate_macd(closing_prices)

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)[-1]

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) >= 0, volumes[1:], -volumes[1:]))
    volume_avg_5 = np.mean(volumes[-5:])
    volume_avg_20 = np.mean(volumes[-20:])
    
    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / (np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) * 100 + 1e-10)
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]**2
    price_position = (closing_prices[-1] - min(closing_prices)) / (max(closing_prices) - min(closing_prices) + 1e-10)
    volume_ratio_regime = volume_avg_5 / (volume_avg_20 + 1e-10)

    # Combine all features
    enhanced_s = np.concatenate([
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        volumes,
        adjusted_closing_prices,
        [sma_5[-1], sma_10[-1], sma_20[-1],
         rsi_5[-1], rsi_10[-1],
         macd_line[-1], signal_line[-1], macd_hist[-1],
         historical_volatility_5, atr_14,
         obv[-1], volume_avg_5, volume_avg_20,
         volatility_ratio, trend_strength, price_position, volume_ratio_regime
        ]
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]  # Recent closing prices
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100  # Historical volatility

    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Large downtrend
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Price drop
            reward -= 50
        if enhanced_s[-2] > 0:  # Trend indicator, could be any trend feature
            reward += 10  # Reward holding in an uptrend
        else:  # If the market is choppy or uncertain
            reward -= 10

    return np.clip(reward, -100, 100)  # Limit reward range