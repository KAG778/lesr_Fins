import numpy as np

def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    # Start with the original state and regime vector
    enhanced = np.concatenate([s, regime_vector])  # 125 dimensions base
    
    # Initialize new features list
    new_features = []

    # Extract closing prices for calculations
    closing_prices = s[0:20]
    
    # Handle edge cases to avoid division by zero
    if len(closing_prices) > 1:
        price_change = closing_prices[-1] - closing_prices[-2]  # Last price - Second last price
        price_average = np.mean(closing_prices)
        
        # Trend-following features
        if abs(trend_strength) > 0.3:
            # Moving Average Crossover Distance (short-term vs long-term)
            short_ma = np.mean(closing_prices[-5:])  # Last 5 days
            long_ma = np.mean(closing_prices[-20:])  # Last 20 days
            new_features.append(short_ma - long_ma)

            # ADX (Average Directional Index) approximation
            directional_movement = (closing_prices[-1] - closing_prices[-2]) / price_average
            new_features.append(np.abs(directional_movement))

            # Trend consistency (simple measure)
            trend_consistency = np.sum(closing_prices[-3:] - closing_prices[-4:-1])  # Last 3 days
            new_features.append(trend_consistency)

        # Mean-reversion features
        else:
            # Bollinger %B
            std_dev = np.std(closing_prices)
            bb_upper = np.mean(closing_prices) + 2 * std_dev
            bb_lower = np.mean(closing_prices) - 2 * std_dev
            bb_percent_b = (closing_prices[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0
            new_features.append(bb_percent_b)

            # RSI (Relative Strength Index)
            gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
            losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) > 14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) > 14 else 0
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            new_features.append(rsi)

    # Volatility features - High Volatility regime
    if volatility_regime > 0.7:
        # Average True Range (ATR)
        high_prices = s[40:60]
        low_prices = s[60:80]
        tr = np.maximum(high_prices[1:] - low_prices[1:], high_prices[1:] - closing_prices[:-1], closing_prices[:-1] - low_prices[1:])
        atr = np.mean(tr[-14:]) if len(tr) > 14 else 0
        new_features.append(atr)

    # Add new features to enhanced state
    return np.concatenate([enhanced, new_features])

def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # Crisis condition override
    if crisis_signal > 0.5:
        return -100.0  # Strong negative in crisis
    
    # Reward logic based on regimes
    if trend_strength > 0.3:  # Strong trend upwards
        reward += 50 if enhanced_s[0] < enhanced_s[1] else -10  # Adjusted based on price direction
    elif trend_strength < -0.3:  # Strong downtrend
        reward += -10  # Cautious reward for downtrend
    else:  # Sideways market
        if enhanced_s[4] > 0.5:  # Mean-reversion opportunity
            reward += 10  # Mild positive reward for mean-reversion

    # Adjust reward for volatility regime
    if regime_vector[1] > 0.7:  # High volatility
        reward *= 0.5  # Reduce reward magnitude
    
    return reward