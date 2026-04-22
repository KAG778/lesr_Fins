import numpy as np

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Calculating Returns
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    
    # Multi-timeframe Trend Indicators
    sma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    sma_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan
    sma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else np.nan
    price_vs_sma_5 = closing_prices[-1] - sma_5 if not np.isnan(sma_5) else np.nan
    price_vs_sma_10 = closing_prices[-1] - sma_10 if not np.isnan(sma_10) else np.nan
    price_vs_sma_20 = closing_prices[-1] - sma_20 if not np.isnan(sma_20) else np.nan

    trend_diff_5_20 = sma_5 - sma_20 if not np.isnan(sma_5) and not np.isnan(sma_20) else np.nan
    
    # Momentum Indicators
    rsi_5 = compute_rsi(returns, period=5)
    rsi_10 = compute_rsi(returns, period=10)
    rsi_14 = compute_rsi(returns, period=14)
    
    macd_line, signal_line = compute_macd(closing_prices)
    macd_histogram = macd_line - signal_line

    # Volatility Indicators
    hist_vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else np.nan
    hist_vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else np.nan
    atr = compute_atr(high_prices, low_prices, closing_prices)
    volatility_ratio = hist_vol_5 / hist_vol_20 if hist_vol_20 != 0 else np.nan

    # Volume-Price Relationship
    obv = compute_obv(closing_prices, volumes)
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan

    # Market Regime Detection
    trend_strength = compute_trend_strength(closing_prices)
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan

    # Creating the enhanced state
    enhanced_s = np.concatenate([
        s,  # Original state
        np.array([sma_5, sma_10, sma_20, price_vs_sma_5, price_vs_sma_10, price_vs_sma_20, trend_diff_5_20,
                  rsi_5, rsi_10, rsi_14, macd_line, signal_line, macd_histogram,
                  hist_vol_5, hist_vol_20, atr, volatility_ratio,
                  obv, volume_ratio, trend_strength, price_position, volume_ratio_regime])
    ])
    
    return enhanced_s

def compute_rsi(returns, period):
    """Compute the Relative Strength Index (RSI) over a given period."""
    if len(returns) < period:
        return np.nan
    gain = np.where(returns > 0, returns, 0)
    loss = np.where(returns < 0, -returns, 0)
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(prices, short_window=12, long_window=26, signal_window=9):
    """Compute the MACD line and signal line."""
    ema_short = np.mean(prices[-short_window:]) if len(prices) >= short_window else np.nan
    ema_long = np.mean(prices[-long_window:]) if len(prices) >= long_window else np.nan
    macd_line = ema_short - ema_long if not np.isnan(ema_short) and not np.isnan(ema_long) else np.nan
    signal_line = np.mean([macd_line])  # Simplified for demonstration
    return macd_line, signal_line

def compute_atr(highs, lows, closes, period=14):
    """Calculate the Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-period:]) if len(tr) >= period else np.nan
    return atr

def compute_obv(prices, volumes):
    """Calculate the On-Balance Volume (OBV)."""
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif prices[i] < prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    return obv[-1]  # Return the last value

def compute_trend_strength(prices):
    """Calculate trend strength using linear regression R²."""
    from sklearn.linear_model import LinearRegression
    x = np.arange(len(prices)).reshape(-1, 1)
    y = prices.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    r_squared = model.score(x, y)
    return r_squared

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 0 else 1  # Avoid division by zero
    threshold = 2 * historical_vol  # Adapt to TSLA's volatility
    
    reward = 0
    
    if position == 0:  # Not holding
        if enhanced_s[-6] > 0 and enhanced_s[-5] < 30:  # Assume SMA and RSI conditions for buy signal
            reward += 50
        elif enhanced_s[-6] < 0:  # If the trend is downward
            reward -= 20
    elif position == 1:  # Holding
        if enhanced_s[-6] < 0:  # If the trend weakens
            reward -= 50
        elif enhanced_s[-6] > 0:  # If the trend is strong
            reward += 30

    # Penalize for recent returns below the adaptive threshold
    if recent_return < -threshold:
        reward -= 50
    
    # Ensure reward is clipped within the defined range
    reward = max(min(reward, 100), -100)
    
    return reward