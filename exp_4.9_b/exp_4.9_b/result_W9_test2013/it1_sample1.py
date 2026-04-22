import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window - 1] = np.mean(prices[:window])  # Start with SMA for the first value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = np.where(deltas < 0, -deltas, 0).mean()
    rs = gain / loss if loss else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else np.nan

def calculate_obv(prices, volumes):
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        obv[i] = obv[i - 1] + volumes[i] if prices[i] > prices[i - 1] else obv[i - 1] - volumes[i]
    return obv

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    enhanced_s = []

    # Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    price_vs_sma_10 = closing_prices[-1] / sma_10 if sma_10 else 0

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]  # Rate of change

    # Volatility Indicators
    historical_vol_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 5 else 0
    historical_vol_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    volatility_ratio = historical_vol_5 / historical_vol_20 if historical_vol_20 else 0

    # Volume Indicators
    obv = calculate_obv(closing_prices, volumes)[-1]  # Last value
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0

    # Market Regime Detection
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] if len(closing_prices) > 1 else 0
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0

    # Construct enhanced state
    enhanced_s.extend([
        sma_5, sma_10, sma_20, ema_5, ema_10, price_vs_sma_10,  # Trend
        rsi_5, rsi_14, momentum,  # Momentum
        historical_vol_5, historical_vol_20, atr, volatility_ratio,  # Volatility
        obv, volume_ratio,  # Volume
        trend_strength, price_position  # Market Regime
    ])

    # Return the enhanced state (original state + new features)
    return np.concatenate((s, enhanced_s))

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Get the position flag (1.0 = holding, 0.0 = not holding)
    closing_prices = enhanced_s[0:20]
    historical_vol = np.std(np.diff(closing_prices)) if len(closing_prices) > 1 else 0
    threshold = 2 * historical_vol  # Use 2x historical volatility as threshold

    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Convert to percentage

    reward = 0

    if position_flag == 0.0:  # Not holding
        if recent_return > threshold and enhanced_s[120] < 30:  # Strong uptrend and oversold
            reward += 50  # Buy signal
        if enhanced_s[120] > 70:  # Overbought
            reward -= 20  # Penalize for potential retracement

    elif position_flag == 1.0:  # Holding
        if recent_return < -threshold:  # Significant drop
            reward -= 50  # Penalty for holding through downturn
        elif enhanced_s[120] > 0:  # If trend is positive
            reward += 20  # Reward for staying in position
        else:
            reward -= 10  # Penalize neutral/negative returns

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds