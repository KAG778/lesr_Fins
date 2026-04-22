import numpy as np

def calculate_sma(prices, period):
    return np.convolve(prices, np.ones(period) / period, mode='valid')

def calculate_ema(prices, period):
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[:period] = np.nan  # Initial values are NaN until EMA can be calculated
    ema[period - 1] = np.mean(prices[:period])  # First EMA value is SMA of the first period
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = -np.where(deltas < 0, deltas, 0)
    avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
    avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]

    # Multi-timeframe Trend Indicators
    sma5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    sma20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else np.nan
    ema5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    ema10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    price_vs_sma = closing_prices[-1] - sma20 if not np.isnan(sma20) else np.nan

    # Momentum Indicators
    rsi5 = calculate_rsi(closing_prices, 5)
    rsi10 = calculate_rsi(closing_prices, 10)
    rsi14 = calculate_rsi(closing_prices, 14)
    macd = calculate_ema(closing_prices, 12)[-1] - calculate_ema(closing_prices, 26)[-1]
    
    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:]) / closing_prices[-6:-1]) if len(closing_prices) > 5 else np.nan
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) if len(closing_prices) > 20 else np.nan
    atr = np.mean(np.max(np.array([high_prices[-1] - low_prices[-1], 
                                    high_prices[-1] - closing_prices[-2], 
                                    closing_prices[-2] - low_prices[-1]]), axis=0)) if len(high_prices) > 1 else np.nan

    # Volume-Price Relationship
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:])) if len(volumes) > 1 else 0
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan
    volume_price_correlation = np.corrcoef(volumes, closing_prices)[0, 1] if len(volumes) > 1 else np.nan

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 > 0 else np.nan
    trend_strength = 1  # Placeholder for R² calculation (needs implementation)
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan

    enhanced_s = np.concatenate((
        s,
        [sma5, sma10, sma20, ema5, ema10, price_vs_sma,
         rsi5, rsi10, rsi14, macd,
         historical_volatility_5, historical_volatility_20, atr,
         obv, volume_ratio, volume_price_correlation,
         volatility_ratio, trend_strength, price_position, volume_ratio_regime]
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)
    threshold = 2 * historical_vol if historical_vol > 0 else 0

    reward = 0

    if position_flag == 0.0:  # Not holding
        if recent_return > threshold:
            reward += 50  # Strong buy signal
        elif recent_return < -threshold:
            reward -= 20  # Potential sell signal
    else:  # Holding
        if recent_return > threshold:
            reward += 30  # Positive trend, reinforce holding
        elif recent_return < -threshold:
            reward -= 50  # Sell signal if trend weakens

    # Penalize uncertain/choppy market conditions
    if np.abs(recent_return) < 0.5 * threshold:
        reward -= 10  # Penalize small moves in volatile environments

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range