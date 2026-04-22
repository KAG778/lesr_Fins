import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-period:]) if len(tr) >= period else np.nan

def calculate_volatility(prices):
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * 100  # Convert to percentage

def calculate_obv(prices, volumes):
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif prices[i] < prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv[-1]  # Return the last value

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
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily momentum

    # Volatility Indicators
    historical_vol_5 = calculate_volatility(closing_prices[-5:])
    historical_vol_20 = calculate_volatility(closing_prices[-20:])
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume Indicators
    obv = calculate_obv(closing_prices, volumes)
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0

    # Market Regime Detection
    volatility_ratio = historical_vol_5 / historical_vol_20 if historical_vol_20 > 0 else 0
    trend_strength = np.corrcoef(np.arange(len(closing_prices)), closing_prices)[0, 1]  # Trend strength using correlation
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0

    # Compile the enhanced state
    enhanced_s = np.concatenate((
        s,
        np.array([
            sma_5 if sma_5 is not None else 0,
            sma_10 if sma_10 is not None else 0,
            sma_20 if sma_20 is not None else 0,
            ema_5 if ema_5 is not None else 0,
            ema_10 if ema_10 is not None else 0,
            rsi_5,
            rsi_10,
            rsi_14,
            momentum,
            historical_vol_5,
            historical_vol_20,
            atr,
            obv,
            volume_ratio,
            volatility_ratio,
            trend_strength,
            price_position,
            volume_ratio_regime
        ])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100  # Daily returns calculation
    recent_return = returns[-1] if len(returns) > 0 else 0

    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 0 else 1  # Avoid division by zero
    threshold = 2 * historical_vol  # Adaptive threshold based on historical volatility

    reward = 0

    if position_flag == 0:  # Not holding
        if enhanced_s[125] < 30:  # RSI 14
            reward += 50  # Potential BUY signal for oversold
        if enhanced_s[124] > 0.1:  # Trend strength
            reward += 10  # Strong uptrend signal
    elif position_flag == 1:  # Holding
        if recent_return < -threshold:  # Weakening trend, consider selling
            reward -= 50
        elif recent_return > 0:  # Positive return, encourage holding
            reward += 30
        if enhanced_s[125] > 70:  # RSI 14
            reward -= 20  # Overbought condition

    # Penalize for recent returns below the adaptive threshold
    if recent_return < -threshold:
        reward -= 50

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]