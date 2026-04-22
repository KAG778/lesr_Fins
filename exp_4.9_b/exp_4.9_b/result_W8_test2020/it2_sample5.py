import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain[-window:]) > 0 else 0
    avg_loss = np.mean(loss[-window:]) if len(loss[-window:]) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def calculate_volatility(returns, window):
    return np.std(returns[-window:]) * 100 if len(returns) >= window else np.nan

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    momentum_5 = closing_prices[-1] - closing_prices[-6]  # 5-day momentum
    momentum_10 = closing_prices[-1] - closing_prices[-11]  # 10-day momentum

    # Volatility Indicators
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = calculate_volatility(daily_returns, 20)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume Indicators
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    volume_avg_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.nan
    volume_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan
    volume_ratio = volumes[-1] / volume_avg_20 if volume_avg_20 > 0 else np.nan

    # Market Regime Features
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    volatility_ratio = historical_volatility / (calculate_volatility(daily_returns, 5) if len(daily_returns) >= 5 else np.nan)
    trend_strength = np.corrcoef(closing_prices[-20:], range(len(closing_prices[-20:])))[:-1, -1][0] if len(closing_prices) >= 20 else np.nan
    volume_ratio_regime = volume_avg_5 / volume_avg_20 if volume_avg_20 > 0 else np.nan

    # Enhanced state
    enhanced_s = np.concatenate((
        s,
        [sma_5[-1] if len(sma_5) > 0 else 0,
         sma_10[-1] if len(sma_10) > 0 else 0,
         sma_20[-1] if len(sma_20) > 0 else 0,
         ema_5[-1] if len(ema_5) > 0 else 0,
         ema_10[-1] if len(ema_10) > 0 else 0,
         rsi_5,
         rsi_14,
         historical_volatility,
         atr_14,
         obv[-1],
         volume_ratio,
         price_position,
         volatility_ratio,
         trend_strength,
         momentum_5,
         momentum_10,
         np.std(daily_returns[-5:]) if len(daily_returns) >= 5 else 0,  # 5-day return volatility
         np.mean(daily_returns[-5:]) if len(daily_returns) >= 5 else 0,  # 5-day average return
         np.mean(volumes[-5:]) if len(volumes) >= 5 else 0  # 5-day average volume
        ]
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    recent_return = daily_returns[-1] if len(daily_returns) > 0 else 0

    # Calculate historical volatility for adaptive threshold
    historical_volatility = np.std(daily_returns) * 100 if len(daily_returns) > 0 else 0
    threshold = 2 * historical_volatility  # Volatility-adaptive threshold

    reward = 0

    if position == 0.0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Sharp decline
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Trend weakening; consider selling
            reward -= 50
        elif recent_return > threshold:  # Continue to hold in uptrend
            reward += 25

    # Penalize uncertain/choppy market conditions
    if np.abs(recent_return) < (0.5 * threshold):  # Choppy condition
        reward -= 10

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds