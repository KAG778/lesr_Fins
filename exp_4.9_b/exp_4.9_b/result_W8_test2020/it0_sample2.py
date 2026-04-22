import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])  # Using simple mean as a placeholder for EMA calculation

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.abs(highs[-window:] - closes[-window:]), 
                    np.abs(lows[-window:] - closes[-window:]))
    return np.mean(tr)

def calculate_volatility(returns, window):
    if len(returns) < window:
        return np.nan
    return np.std(returns[-window:])

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # A. Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    price_sma_5 = closing_prices[-1] - sma_5 if not np.isnan(sma_5) else np.nan
    price_sma_10 = closing_prices[-1] - sma_10 if not np.isnan(sma_10) else np.nan
    price_sma_20 = closing_prices[-1] - sma_20 if not np.isnan(sma_20) else np.nan

    # B. Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # C. Volatility Indicators
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_vol_5 = calculate_volatility(returns, 5)
    historical_vol_20 = calculate_volatility(returns, 20)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # D. Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))  # On-Balance Volume
    volume_avg_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.nan
    volume_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan

    # E. Market Regime Detection
    volatility_ratio = historical_vol_5 / historical_vol_20 if historical_vol_20 > 0 else np.nan
    trend_strength = np.nan  # Placeholder for R² calculation
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    volume_ratio_regime = volume_avg_5 / volume_avg_20 if volume_avg_20 > 0 else np.nan

    # Create the enhanced state
    enhanced_s = np.concatenate([
        s,
        [sma_5, sma_10, sma_20, price_sma_5, price_sma_10, price_sma_20,
         rsi_5, rsi_10, rsi_14,
         historical_vol_5, historical_vol_20, atr,
         obv[-1], volume_avg_5, volume_avg_20,
         volatility_ratio, trend_strength, price_position, volume_ratio_regime]
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    recent_return = returns[-1] if len(returns) > 0 else 0

    # Calculate historical volatility for adaptive threshold
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    if position_flag == 0.0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Sharp decline
            reward -= 50
    else:  # Holding
        if recent_return > threshold:  # Continue to hold in uptrend
            reward += 25
        elif recent_return < -threshold:  # Trend weakening; consider selling
            reward -= 50

    # Penalize uncertain/choppy market conditions
    if np.abs(recent_return) < (0.5 * threshold):  # Choppy condition
        reward -= 10

    return np.clip(reward, -100, 100)