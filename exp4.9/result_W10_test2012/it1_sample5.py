import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Calculate closing prices for the last 20 days
    closing_prices = s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]

    # Historical volatility for adaptive thresholds
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility

    # Features based on market regimes
    if abs(trend_strength) > 0.3:  # TREND regimes
        short_ma = np.mean(closing_prices[-5:])  # Last 5 days MA
        long_ma = np.mean(closing_prices)  # Last 20 days MA
        ma_crossover = short_ma - long_ma
        new_features.append(ma_crossover)  # Moving Average Crossover
        new_features.append(historical_volatility)  # Historical volatility
        new_features.append(np.mean(daily_returns[-5:]))  # Average return over last 5 days

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regimes
        upper_band = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_band = np.mean(closing_prices) - 2 * np.std(closing_prices)
        pct_b = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0
        new_features.append(pct_b)  # Bollinger %B
        new_features.append(np.mean(daily_returns[-5:]))  # Average return over last 5 days
        new_features.append(np.max(closing_prices) - np.min(closing_prices))  # Range width

    # Append new features to the enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Different reward logic per regime
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0 and enhanced_s[0] > enhanced_s[1]:  # Uptrend
            reward += 30.0  # Positive for aligning with trend
        elif trend_strength < 0 and enhanced_s[0] < enhanced_s[1]:  # Downtrend
            reward -= 20.0  # Negative for aligning against trend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if enhanced_s[125] < 0.5:  # Assuming Bollinger %B is at index 125
            reward += 15.0  # Mild positive for mean-reversion opportunity
        elif enhanced_s[125] > 0.5:  # If Bollinger %B is high
            reward -= 10.0  # Negative for chasing breakouts

    # Adjust reward based on volatility
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down rewards in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within range