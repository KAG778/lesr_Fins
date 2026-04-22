import numpy as np

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
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    if len(prices) < 26:
        return (np.nan, np.nan, np.nan)
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema(prices, 9)  # Signal line is EMA of MACD
    histogram = macd_line - signal_line
    return (macd_line, signal_line, histogram)

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.maximum(np.abs(highs[-window:] - closes[-window:]), 
                               np.abs(lows[-window:] - closes[-window:])))
    return np.mean(tr)

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    macd_line, signal_line, histogram = calculate_macd(closing_prices)

    # Volatility Indicators
    atr_14 = calculate_atr(high_prices, lows, closing_prices, 14)
    historical_volatility_5 = calculate_volatility(closing_prices, 5)
    historical_volatility_20 = calculate_volatility(closing_prices, 20)
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else np.nan

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))[-1]  # On-Balance Volume
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Market Regime Detection
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2 if len(closing_prices) > 1 else np.nan
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Create the enhanced state with new features
    enhanced_s = np.concatenate([
        s,
        [sma_5, sma_10, sma_20, ema_5, ema_10,
         rsi_5, rsi_14, macd_line, signal_line, histogram,
         atr_14, historical_volatility_5, historical_volatility_20, volatility_ratio,
         obv, volume_ratio, trend_strength, price_position, volume_ratio_regime]
    ])
    
    # Ensure the enhanced state is 140-150 dimensional
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility_5 = calculate_volatility(closing_prices, 5)
    threshold = 2 * historical_volatility_5 if historical_volatility_5 is not np.nan else 0

    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold and enhanced_s[24] > 70:  # RSI and return condition for buying
            reward += 50  # Strong buy signal
        elif recent_return < -threshold:  # Negative signal
            reward -= 50  # Strong downtrend
    elif position_flag == 1:  # Holding
        if recent_return < -threshold:  # Sell signal
            reward -= 50
        elif recent_return > 0:  # Reward for holding during slight uptrend
            reward += 20

    # Penalize for uncertain/choppy market conditions
    if np.isnan(threshold) or threshold < 1:  # Arbitrary threshold for "uncertain"
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]