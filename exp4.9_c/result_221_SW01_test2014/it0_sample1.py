import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    num_days = len(closing_prices)

    # Feature 1: Moving Average over the last 5 days
    if num_days >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = np.nan  # Handle edge case

    # Feature 2: Relative Strength Index (RSI) calculation
    deltas = np.diff(closing_prices)  # Price change
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.nan
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.nan
    
    # Avoid division by zero
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # Feature 3: Volatility (standard deviation of closing prices)
    volatility = np.std(closing_prices[-5:]) if num_days >= 5 else np.nan

    # Return new features as a numpy array
    features = [moving_average, rsi, volatility]
    return np.array(features)


def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Random strong penalty
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)  # Moderate penalty

    # Check if risk is acceptable for further evaluation
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:
                reward += 10  # Positive reward for upward trend
            else:
                reward += 10  # Positive reward for downward trend

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Assuming features include information about overbought/oversold
            # Here we should have some logic based on computed features
            reward += 5  # Placeholder for mean-reversion reward

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)