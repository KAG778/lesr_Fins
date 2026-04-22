import numpy as np

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # A. Multi-timeframe Trend Indicators
    sma_5 = np.mean(closing_prices[-5:])  # 5-day SMA
    sma_10 = np.mean(closing_prices[-10:])  # 10-day SMA
    sma_20 = np.mean(closing_prices[-20:])  # 20-day SMA
    ema_5 = np.mean(closing_prices[-5:])  # 5-day EMA (approximation)
    ema_10 = np.mean(closing_prices[-10:])  # 10-day EMA (approximation)
    ema_20 = np.mean(closing_prices[-20:])  # 20-day EMA (approximation)

    price_vs_sma_5 = closing_prices[-1] - sma_5
    price_vs_sma_10 = closing_prices[-1] - sma_10
    price_vs_sma_20 = closing_prices[-1] - sma_20

    trend_features = np.array([sma_5, sma_10, sma_20, price_vs_sma_5, price_vs_sma_10, price_vs_sma_20])

    # B. Momentum Indicators
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    rsi_5 = np.clip((np.sum(returns[-5:]) / 5) * 100, 0, 100)  # Simplified RSI for example
    rsi_10 = np.clip((np.sum(returns[-10:]) / 10) * 100, 0, 100)
    rsi_14 = np.clip((np.sum(returns[-14:]) / 14) * 100, 0, 100)

    macd_line = np.mean(closing_prices[-12:]) - np.mean(closing_prices[-26:])
    signal_line = np.mean(macd_line)  # Simplified
    macd_histogram = macd_line - signal_line

    momentum_features = np.array([rsi_5, rsi_10, rsi_14, macd_line, signal_line, macd_histogram])

    # C. Volatility Indicators
    daily_volatility_5 = np.std(returns[-5:]) * 100
    daily_volatility_20 = np.std(returns[-20:]) * 100
    atr = np.mean(high_prices[-1] - low_prices[-1])  # Simplified ATR
    volatility_ratio = daily_volatility_5 / daily_volatility_20 if daily_volatility_20 != 0 else 0

    volatility_features = np.array([daily_volatility_5, daily_volatility_20, atr, volatility_ratio])

    # D. Volume-Price Relationship
    obv = np.sum(np.sign(np.diff(closing_prices) * volumes[1:]))  # Simplified OBV
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else 0

    volume_features = np.array([obv, volume_ratio])

    # E. Market Regime Detection
    volatility_ratio_market = daily_volatility_5 / daily_volatility_20 if daily_volatility_20 != 0 else 0
    trend_strength = 1.0  # Dummy value for trend strength (should be calculated based on linear regression)
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else 0

    regime_features = np.array([volatility_ratio_market, trend_strength, price_position, volume_ratio_regime])

    # Combine all features
    enhanced_s = np.concatenate((s, trend_features, momentum_features, volatility_features, volume_features, regime_features))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1]
    
    recent_return = returns[-1] * 100  # Convert to percentage
    historical_volatility = np.std(returns) * 100  # Historical volatility in percentage
    threshold = 2 * historical_volatility  # Relative threshold based on historical volatility
    
    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend signal
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Weakening trend
            reward -= 50
        elif recent_return > 0:  # Maintain position during uptrend
            reward += 20

    # Penalize uncertain/choppy market conditions (e.g., low trend strength)
    trend_strength = enhanced_s[-5]  # Assuming trend_strength is one of the last features
    if trend_strength < 0.5:  # Arbitrary threshold for choppy market
        reward -= 30

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]