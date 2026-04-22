import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(prices.shape)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:]) if len(gains) >= window else 0
    avg_loss = np.mean(losses[-window:]) if len(losses) >= window else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-period:])

def calculate_obv(prices, volumes):
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif prices[i] < prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    return obv

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else 0
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else 0
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else 0
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else 0
    ema_10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else 0

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5) if len(closing_prices) >= 5 else np.nan
    rsi_14 = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) >= 2 else np.nan

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices)) * 100 if len(closing_prices) > 1 else np.nan
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) * 100 if len(closing_prices) >= 20 else np.nan
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume-Price Relationship
    obv = calculate_obv(closing_prices, volumes)
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 > 0 else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] if len(closing_prices) > 1 else np.nan
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Compile the enhanced state with new features
    enhanced_s = np.concatenate((
        s,
        np.array([
            sma_5, sma_10, sma_20,
            ema_5, ema_10,
            rsi_5, rsi_14,
            historical_volatility_5, historical_volatility_20, atr,
            obv, volume_ratio,
            volatility_ratio, trend_strength, price_position, volume_ratio_regime,
            momentum,
            closing_prices[-1] / sma_5 if sma_5 > 0 else 0,
            closing_prices[-1] / sma_10 if sma_10 > 0 else 0,
            closing_prices[-1] / sma_20 if sma_20 > 0 else 0
        ])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return
    historical_volatility = np.std(np.diff(closing_prices)) * 100 if len(closing_prices) > 1 else 1  # Avoid division by zero

    # Use volatility-adaptive thresholds
    threshold = 2 * historical_volatility  # Adaptable threshold based on volatility

    reward = 0
    
    if position == 0:  # Not holding
        if enhanced_s[15] < 30:  # RSI condition for buy signal (oversold)
            reward += 50
        if enhanced_s[12] > 0.1:  # Trend strength condition for buy signal
            reward += 10
    else:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        elif recent_return > 0:  # Positive return
            reward += 30
        if enhanced_s[15] > 70:  # RSI condition for sell signal (overbought)
            reward -= 10
        if enhanced_s[12] < 0.1:  # Weak trend signal
            reward -= 10
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]