import numpy as np

def revise_state(s):
    # Extract closing prices and volumes from the raw state
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate returns
    returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # A. Multi-timeframe Trend Indicators
    sma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    sma_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan
    sma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else np.nan
    ema_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    ema_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan
    price_vs_sma_20 = closing_prices[-1] / sma_20 if sma_20 else np.nan
    price_vs_ema_10 = closing_prices[-1] / ema_10 if ema_10 else np.nan
    
    trend_indicators = [sma_5, sma_10, sma_20, ema_5, ema_10, price_vs_sma_20, price_vs_ema_10]

    # B. Momentum Indicators
    rsi_5 = 100 - (100 / (1 + np.mean(returns[-5:])) if len(returns) >= 5 else np.nan)
    rsi_10 = 100 - (100 / (1 + np.mean(returns[-10:])) if len(returns) >= 10 else np.nan)
    macd_line = np.mean(closing_prices[-12:]) - np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else np.nan
    macd_signal = np.mean(closing_prices[-9:]) if len(closing_prices) >= 9 else np.nan
    
    momentum_indicators = [rsi_5, rsi_10, macd_line, macd_signal]

    # C. Volatility Indicators
    historical_volatility_5 = np.std(returns[-5:]) if len(returns) >= 5 else np.nan
    historical_volatility_20 = np.std(returns[-20:]) if len(returns) >= 20 else np.nan
    atr = np.mean(high_prices[-20:]) - np.mean(low_prices[-20:]) if len(high_prices) >= 20 else np.nan
    
    volatility_indicators = [historical_volatility_5, historical_volatility_20, atr]

    # D. Volume-Price Relationship
    obv = np.sum(np.where(returns > 0, volumes[1:], -volumes[1:])) if len(volumes) > 1 else np.nan
    volume_price_corr = np.corrcoef(volumes, closing_prices)[0, 1] if len(volumes) > 1 else np.nan
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan
    
    volume_indicators = [obv, volume_price_corr, volume_ratio]

    # E. Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] if len(closing_prices) > 1 else np.nan
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if len(closing_prices) >= 20 else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan

    market_regime_indicators = [volatility_ratio, trend_strength, price_position, volume_ratio_regime]

    # Combine all features
    enhanced_s = np.concatenate([s, trend_indicators, momentum_indicators, volatility_indicators, volume_indicators, market_regime_indicators])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Get the position flag (1.0 = holding, 0.0 = not holding)
    
    # Historical volatility
    returns = np.diff(enhanced_s[0:20]) / enhanced_s[0:19]
    historical_vol = np.std(returns) * 100  # Convert to percentage
    threshold = 2 * historical_vol  # Use 2x historical volatility as threshold
    
    reward = 0

    # Reward logic based on position flag
    if position_flag == 0.0:  # Not holding
        # Check for strong uptrend (using trend_strength)
        if enhanced_s[120 + 6] >= 0.8:  # Assuming trend_strength is at index 126
            reward += 50  # Strong uptrend
        if enhanced_s[120 + 0] < 30:  # Assuming rsi_5 is at index 120 + 0
            reward += 30  # Oversold condition
        
    elif position_flag == 1.0:  # Holding
        # Check for maintaining position during uptrend
        if enhanced_s[120 + 6] >= 0.8:  # Strong uptrend
            reward += 20  # Continue holding
        if enhanced_s[120 + 0] < 30:  # Oversold condition
            reward += 10  # Positive reward for holding during oversold
        
        # Check for weak trend to consider selling
        if enhanced_s[120 + 6] < 0.5:  # Weak trend
            reward -= 30  # Consider selling
            
    return reward