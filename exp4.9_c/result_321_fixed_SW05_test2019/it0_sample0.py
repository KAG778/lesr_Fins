import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Prepare features from the raw OHLCV data
    closing_prices = s[::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Momentum (5-day momentum)
    momentum_days = 5
    price_momentum = closing_prices[momentum_days:] - closing_prices[:-momentum_days]
    price_momentum = np.concatenate((np.zeros(momentum_days), price_momentum))  # Padding for alignment

    # Feature 2: Volatility (standard deviation of returns over the last 5 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.concatenate((
        np.zeros(5),  # Padding for the first 5 days
        np.array([np.std(returns[i:i + 5]) for i in range(len(returns) - 5 + 1)])
    ))

    # Feature 3: Volume Change (percentage change)
    volume_change = np.concatenate((
        np.zeros(1),  # No change for the first day
        (trading_volumes[1:] - trading_volumes[:-1]) / trading_volumes[:-1]
    ))

    # Combine features into a single array
    features = np.array([
        price_momentum[-20:],  # last 20 days momentum
        volatility[-20:],      # last 20 days volatility
        volume_change[-20:]    # last 20 days volume change
    ]).flatten()

    # Return only the new computed features
    return features

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

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 10.0 * features[1]  # MILD POSITIVE for SELL-aligned features (based on volatility)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive momentum
            reward += trend_direction * features[0] * 10.0  # Strong reward for aligning with trend
        elif features[0] < 0:  # Negative momentum
            reward += trend_direction * features[0] * 10.0  # Reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # mild positive for mean-reversion BUY
        elif features[0] > 0:  # Overbought condition
            reward -= 5.0  # Penalize breakout-chasing SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))