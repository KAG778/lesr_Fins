import numpy as np

def compute_moving_average(prices, period):
    """Compute moving average for the given period."""
    return np.mean(prices[-period:]) if len(prices) >= period else np.nan

def compute_atr(prices, high_prices, low_prices, period):
    """Compute Average True Range (ATR)."""
    tr = np.maximum(high_prices[-1] - low_prices[-1], 
                    np.maximum(abs(high_prices[-1] - prices[-2]), 
                               abs(low_prices[-1] - prices[-2])))
    return np.mean([tr]) if len(prices) >= period else np.nan

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []
    
    # Calculate recent closing prices for features
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Trend-following features
    if abs(trend_strength) > 0.3:
        short_ma = compute_moving_average(closing_prices, 5)
        long_ma = compute_moving_average(closing_prices, 20)
        ma_crossover_distance = short_ma - long_ma
        new_features.append(ma_crossover_distance)
        
        # Average True Range
        atr = compute_atr(closing_prices, high_prices, low_prices, 14)
        new_features.append(atr)
        
        # Trend consistency (simple measure of price increases)
        trend_consistency = np.sum(np.diff(closing_prices) > 0) / 19  # Normalized by days
        new_features.append(trend_consistency)

    # Mean-reversion features
    else:
        bollinger_mid = np.mean(closing_prices[-20:])
        bollinger_std = np.std(closing_prices[-20:])
        bollinger_upper = bollinger_mid + 2 * bollinger_std
        bollinger_lower = bollinger_mid - 2 * bollinger_std
        bollinger_percent_b = (closing_prices[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        new_features.append(bollinger_percent_b)
        
        # RSI (Relative Strength Index)
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
        rsi = 100 - (100 / (1 + rs)) if not np.isnan(rs) else np.nan
        new_features.append(rsi)

    # Append new features to enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Reward logic per regime
    if trend_strength > 0.3:  # Strong uptrend
        if regime_vector[2] > 0:  # Positive momentum
            reward += 20  # Positive reward for trend alignment
        else:
            reward -= 10  # Cautious reward for negative momentum
            
    elif trend_strength < -0.3:  # Strong downtrend
        reward -= 10  # Caution on buying in downtrend
    
    elif abs(trend_strength) < 0.15:  # Sideways market
        if regime_vector[3] < 0:  # Potential for mean reversion
            reward += 10  # Mild positive for counter-trend opportunities
    
    # High volatility adjustment
    if regime_vector[1] > 0.7:  # High volatility
        reward -= 5  # Negative for aggressive entries
    
    return reward