import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    # Define volatility threshold for adapting features
    volatility_threshold = np.std(s[0:20]) * 2  # Example threshold (2 standard deviations)

    # Features for TREND regimes (|trend_strength| > 0.3)
    if abs(trend_strength) > 0.3:
        short_ma = np.mean(s[0:20][-5:])  # Last 5 days closing price
        long_ma = np.mean(s[0:20])  # Last 20 days closing price
        ma_crossover = short_ma - long_ma
        new_features.append(ma_crossover)  # Moving Average Crossover Distance

        # Trend momentum feature (momentum signal)
        momentum = np.mean(np.diff(s[0:20]))  # Average momentum over the last 20 days
        new_features.append(momentum)

        # Historical volatility as a feature
        daily_returns = np.diff(s[0:20]) / s[0:20][:-1]
        historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
        new_features.append(historical_volatility)

    # Features for SIDEWAYS regimes (|trend_strength| < 0.15)
    elif abs(trend_strength) < 0.15:
        upper_bollinger = np.mean(s[0:20]) + 2 * np.std(s[0:20])
        lower_bollinger = np.mean(s[0:20]) - 2 * np.std(s[0:20])
        bollinger_percent_b = (s[0] - lower_bollinger) / (upper_bollinger - lower_bollinger) if (upper_bollinger - lower_bollinger) != 0 else 0
        new_features.append(bollinger_percent_b)  # Bollinger %B

        # Mean-reversion signal
        new_features.append(np.mean(daily_returns[-5:]))  # Average return over last 5 days
        new_features.append(np.max(s[0:20]) - np.min(s[0:20]))  # Price range

    # Ensure new features are concatenated to the enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Reward logic based on different regimes
    if abs(trend_strength) > 0.3:  # TREND regime
        if trend_strength > 0:  # Uptrend
            reward += 50.0  # Strong positive for aligning with trend
        else:  # Downtrend
            reward -= 30.0  # Cautious negative for going against trend

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        if enhanced_s[125] < 0.5:  # Assuming Bollinger %B is at index 125
            reward += 20.0  # Positive for mean-reversion opportunity

    # Adjust reward based on volatility regime
    if regime_vector[1] > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Scale down all rewards

    return np.clip(reward, -100, 100)  # Ensure reward is within range