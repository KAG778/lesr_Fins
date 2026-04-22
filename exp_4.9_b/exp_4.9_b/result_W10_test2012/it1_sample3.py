import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema[-1]

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:]) if len(gains) >= window else 0
    avg_loss = np.mean(losses[-window:]) if len(losses) >= window else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.maximum(np.abs(highs[-window:] - closes[-window-1:-1]), 
                               np.abs(lows[-window:] - closes[-window-1:-1])))
    return np.mean(tr)

def calculate_volatility(prices, window):
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) if len(returns) >= window else np.nan

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Enhanced features
    features = []

    # Trend Indicators
    features.append(calculate_sma(closing_prices, 5))  # SMA 5
    features.append(calculate_sma(closing_prices, 10))  # SMA 10
    features.append(calculate_sma(closing_prices, 20))  # SMA 20
    features.append(calculate_ema(closing_prices, 5))   # EMA 5
    features.append(calculate_ema(closing_prices, 10))  # EMA 10
    features.append(closing_prices[-1] - features[-1])   # Price vs EMA 10
    features.append(calculate_rsi(closing_prices, 14))   # RSI 14

    # Momentum Indicators
    macd_line, signal_line, histogram = calculate_macd(closing_prices)
    features.append(macd_line)  # MACD
    features.append(signal_line) # Signal Line
    features.append(histogram)   # MACD Histogram

    # Volatility Indicators
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    features.append(atr)  # ATR
    historical_volatility_5 = calculate_volatility(closing_prices, 5)
    historical_volatility_20 = calculate_volatility(closing_prices, 20)
    features.append(historical_volatility_5) # 5-day Historical Volatility
    features.append(historical_volatility_20) # 20-day Historical Volatility
    features.append(historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else np.nan)  # Volatility Ratio

    # Volume-Price Relationship
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[:-1], -volumes[:-1]))
    features.append(obv)  # On-Balance Volume
    features.append(volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else np.nan)  # Volume Ratio

    # Market Regime Detection
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2 if len(closing_prices) > 1 else np.nan
    features.append(trend_strength)  # Trend Strength
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    features.append(price_position)  # Price Position
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan
    features.append(volume_ratio_regime)  # Volume Ratio Regime

    # Extend to ensure we have between 120-150 dimensions
    enhanced_s = np.concatenate((s, np.array(features)), axis=0)
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0
    historical_volatility = calculate_volatility(closing_prices, 5)  # Use 5-day for short-term volatility
    threshold = 2 * historical_volatility if historical_volatility > 0 else 1  # Adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong BUY signal
            reward += 50
        elif recent_return < -threshold:  # Strong SELL signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Sell signal in a downturn
            reward -= 50
        elif recent_return > 0:  # Good to HOLD
            reward += 20

    # Penalize for uncertain/choppy market conditions
    if np.isnan(threshold) or threshold < 1:  # Arbitrary threshold for "uncertain"
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]