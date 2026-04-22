import numpy as np

def revise_state(s, regime_vector):
    # Extract regime vector components
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]

    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    new_features = []

    closing_prices = s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility

    if abs(trend_strength) > 0.3:  # TREND regime
        # Trend-following features
        short_ma = np.mean(closing_prices[-5:])  # Last 5 days
        long_ma = np.mean(closing_prices[-20:])  # Last 20 days
        ma_distance = short_ma - long_ma
        
        adx = np.mean(np.abs(daily_returns[-5:]))  # Simplified ADX
        new_features = [ma_distance, adx, historical_volatility]

    elif abs(trend_strength) < 0.15:  # SIDEWAYS regime
        # Mean-reversion features
        upper_bollinger = np.mean(closing_prices) + 2 * np.std(closing_prices)
        lower_bollinger = np.mean(closing_prices) - 2 * np.std(closing_prices)
        
        bollinger_percent_b = (closing_prices[-1] - lower_bollinger) / (upper_bollinger - lower_bollinger + 1e-10)
        rsi = (100 - (100 / (1 + np.mean(daily_returns[-14:]) / np.std(daily_returns[-14:])))
                if np.std(daily_returns[-14:]) > 0 else 0)  # Avoid division by zero
        new_features = [bollinger_percent_b, rsi, np.max(closing_prices) - np.min(closing_prices)]

    # Ensure we return at least 3 new features
    new_features += [0] * (3 - len(new_features))  # Fill with zeros if necessary

    return np.concatenate([enhanced, np.array(new_features)])

def intrinsic_reward(enhanced_s):
    # Extract regime_vector from enhanced_state[120:125]
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]

    reward = 0.0

    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis

    # Reward logic based on different regimes
    if trend_strength > 0.3:  # Strong uptrend (TREND regime)
        reward += 50.0  # Positive reward for trend-aligned actions
        # Penalize counter-trend actions
        if regime_vector[2] < 0:  # Momentum signal indicates downward momentum
            reward -= 20.0  # Penalize aggressive positions against trend

    elif trend_strength < -0.3:  # Strong downtrend (TREND regime)
        reward -= 50.0  # Negative reward for buying in downtrend
        # Encourage short position if momentum is up
        if regime_vector[2] > 0:
            reward += 15.0  # Mild reward for shorts in downtrend

    elif -0.15 <= trend_strength <= 0.15:  # Sideways market (SIDEWAYS regime)
        if regime_vector[3] == -1:  # Potential mean-reversion opportunity
            reward += 20.0  # Positive for counter-trend entries

    # Adjust reward for volatility
    volatility_regime = regime_vector[1]
    if volatility_regime > 0.7:  # HIGH_VOL regime
        reward *= 0.5  # Reduce reward magnitude for high volatility

    return np.clip(reward, -100, 100)  # Ensure the reward is within the specified range