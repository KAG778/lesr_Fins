import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]  # Starting point
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, window):
    """Calculate Average True Range."""
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.mean(tr[-window:])
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate SMAs
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    # Calculate EMAs
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    # Price relative to moving averages
    price_vs_sma_5 = closing_prices[-1] / sma_5[-1] if len(sma_5) > 0 else 0
    price_vs_ema_10 = closing_prices[-1] / ema_10[-1] if len(ema_10) > 0 else 0
    
    # Calculate RSI
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)

    # Calculate ATR
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(returns) * 100  # Convert to percentage

    # Market Regime Features
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_avg_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    volume_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    volume_ratio_regime = volume_avg_5 / volume_avg_20 if volume_avg_20 != 0 else 0
    
    # Prepare enhanced state
    enhanced_s = np.concatenate([
        s,
        np.array([
            sma_5[-1] if len(sma_5) > 0 else 0,
            sma_10[-1] if len(sma_10) > 0 else 0,
            sma_20[-1] if len(sma_20) > 0 else 0,
            ema_5[-1] if len(ema_5) > 0 else 0,
            ema_10[-1] if len(ema_10) > 0 else 0,
            price_vs_sma_5,
            price_vs_ema_10,
            rsi_5,
            rsi_14,
            atr_14,
            historical_volatility,
            price_position,
            volume_ratio_regime,
            historical_volatility / (np.std(np.diff(closing_prices[-20:])) * 100) if len(closing_prices) >= 20 else 0,  # Volatility ratio
            np.corrcoef(volumes, closing_prices)[0, 1] if len(volumes) > 1 else 0,  # Volume-price correlation
            np.mean(volumes[-5:]) / np.mean(volumes) if len(volumes) > 0 else 0  # Volume ratio
        ])
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Recent return in percentage
    historical_volatility = enhanced_s[10]  # Historical volatility
    
    # Define thresholds based on historical volatility
    threshold = 2 * historical_volatility

    reward = 0
    
    if position == 0:  # Not holding
        if (enhanced_s[6] > 1) and (enhanced_s[8] < 30):  # Strong uptrend and oversold
            reward += 50  # Strong buy signal
        elif (enhanced_s[8] > 70):  # Overbought
            reward -= 20  # Consider caution for buying
    else:  # Holding
        if recent_return < -threshold:  # Significant drop
            reward -= 50  # Penalty for holding during a significant drop
        elif enhanced_s[6] < 1:  # Downtrend
            reward += 30  # Reward for selling in downtrend
        else:  # Maintain position
            reward += 10  # Small reward for holding in an uptrend

    # Penalize uncertain market conditions
    if enhanced_s[11] > 0.5:  # Price position near extremes
        reward -= 10  # Penalize choppy conditions
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds