import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    
    # Feature 1: EMA Divergence (20-day period)
    ema = np.mean(closing_prices[-20:])  # Using the mean as a simple EMA approximation.
    price_divergence = (closing_prices[-1] - ema) / (ema + 1e-10)

    # Feature 2: VWAP Position
    total_volume = np.sum(volumes)
    if total_volume > 0:
        vwap = np.sum(closing_prices * volumes) / total_volume
        vwap_position = (closing_prices[-1] - vwap) / (vwap + 1e-10)
    else:
        vwap_position = 0.0

    # Feature 3: Standard Deviation of Returns
    returns = np.diff(closing_prices) / closing_prices[:-1]
    volatility = np.std(returns)

    return np.array([price_divergence, vwap_position, volatility])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50.0  # Strong negative for BUY-aligned features
        reward += 10.0 * (features[1] < 0)  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 25.0  # Reward for following trend based on EMA divergence

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 10.0  # Encourage BUY
        elif features[0] > 0.01:  # Overbought condition
            reward += 10.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))