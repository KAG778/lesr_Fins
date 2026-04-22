import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

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
    return np.mean(tr[-window:])

def calculate_volatility(returns):
    return np.std(returns) * 100

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    ema_10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5) if len(closing_prices) >= 5 else np.nan
    rsi_14 = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan

    # Volatility Indicators
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_vol = calculate_volatility(daily_returns) if len(daily_returns) > 0 else np.nan
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14) if len(high_prices) >= 14 else np.nan

    # Volume Indicators
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))  # On-Balance Volume
    volume_avg_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.nan
    volume_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan
    volume_ratio = volumes[-1] / volume_avg_20 if volume_avg_20 > 0 else np.nan

    # Market Regime Features
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    volatility_ratio = historical_vol / (calculate_volatility(daily_returns[-20:]) if len(daily_returns) >= 20 else np.nan) if len(daily_returns) >= 20 else np.nan
    trend_strength = np.corrcoef(closing_prices[-20:], range(len(closing_prices[-20:])))[:-1, -1][0] if len(closing_prices) >= 20 else np.nan

    # New features
    enhanced_s = np.array([
        sma_5, sma_10, ema_5, ema_10,
        rsi_5, rsi_14, historical_vol, atr,
        obv[-1], volume_avg_5, volume_avg_20,
        price_position, volatility_ratio, trend_strength,
        daily_returns[-1] if len(daily_returns) > 0 else np.nan,  # Latest return
        np.mean(daily_returns[-5:]) if len(daily_returns) >= 5 else np.nan,  # Average return last 5 days
        np.mean(daily_returns[-10:]) if len(daily_returns) >= 10 else np.nan,  # Average return last 10 days
        np.std(daily_returns[-5:]) if len(daily_returns) >= 5 else np.nan,  # Volatility last 5 days
        np.std(daily_returns[-10:]) if len(daily_returns) >= 10 else np.nan,  # Volatility last 10 days
        volume_ratio
    ])

    # Combine with original state
    enhanced_s = np.concatenate([s, enhanced_s])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    recent_return = daily_returns[-1] if len(daily_returns) > 0 else 0

    # Calculate historical volatility for adaptive threshold
    historical_vol = np.std(daily_returns) * 100 if len(daily_returns) > 0 else 0
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    if position == 0:  # Not holding
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