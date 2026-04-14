import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    closing_prices = s[0::6]
    opening_prices = s[1::6]
    high_prices = s[2::6]
    low_prices = s[3::6]
    trading_volumes = s[4::6]

    # Feature 1: Daily Return Percentage
    daily_returns = np.zeros(len(closing_prices) - 1)
    for i in range(1, len(closing_prices)):
        if closing_prices[i-1] != 0:
            daily_returns[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Feature 2: Average Trading Volume over the last 20 days
    avg_volume = np.mean(trading_volumes)
    
    # Feature 3: Price Volatility (Standard Deviation of Daily Returns)
    price_volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0.0

    features = [
        np.mean(daily_returns),  # Mean daily return
        avg_volume,              # Average volume
        price_volatility         # Price volatility
    ]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management (deterministic, NO random calls)
    if risk_level > 0.7:
        # Strong negative penalty for buying in dangerous conditions
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        # Moderate negative penalty for buying in elevated risk
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            # Positive reward for aligning with upward trend
            reward += features[0] * 20.0  # Positive reward proportional to mean daily return
        else:
            # Positive reward for aligning with downward trend
            reward += -features[0] * 20.0  # Positive reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 10.0  # Buy signal
        elif features[0] > 0:  # Overbought condition
            reward += 5.0  # Sell signal
        else:
            reward += 0.0  # Neutral

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in uncertain markets

    return float(np.clip(reward, -100, 100))