import numpy as np

def compute_volatility(prices):
    """Compute historical volatility based on closing prices."""
    returns = np.diff(prices) / prices[:-1]  # Daily returns
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    new_features = []

    closing_prices = s[0:20]
    historical_volatility = compute_volatility(closing_prices)

    # Feature extraction based on regime
    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Short-term MA
        long_ma = np.mean(closing_prices[-20:])  # Long-term MA
        ma_crossover_distance = short_ma - long_ma
        
        # Trend consistency
        returns = np.diff(closing_prices) / closing_prices[:-1]
        trend_consistency = np.std(returns)
        
        new_features = [ma_crossover_distance, trend_consistency, historical_volatility]
        
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        
        bollinger_percent_b = (closing_prices[-1] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std) if rolling_std != 0 else 0
        
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100
        
        new_features = [bollinger_percent_b, rsi, historical_volatility]
    
    # Always include historical volatility
    new_features.append(historical_volatility)

    return np.concatenate([enhanced, new_features])


def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    volatility_regime = regime_vector[1]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Assess reward based on regimes
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0:
            reward += 50.0  # Strong uptrend reward
        else:
            reward -= 20.0  # Cautious reward in downtrend
    
    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        bollinger_percent_b = enhanced_s[125]  # Assuming this is the Bollinger %B feature
        if bollinger_percent_b < 0.5:
            reward += 10.0  # Positive for mean-reversion potential

    # Volatility adjustment
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward -= 30.0  # Penalize aggressive entries in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is in range [-100, 100]