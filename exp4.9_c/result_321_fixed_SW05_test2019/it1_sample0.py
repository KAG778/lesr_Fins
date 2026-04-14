import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    high_prices = s[2::6]      # Extract high prices
    low_prices = s[3::6]       # Extract low prices

    features = []
    
    # Feature 1: 5-Day Price Momentum (Current close vs close 5 days ago)
    momentum_days = 5
    price_momentum = closing_prices[-1] - closing_prices[-(momentum_days + 1)] if len(closing_prices) > momentum_days else 0
    features.append(price_momentum)

    # Feature 2: 5-Day Volatility (standard deviation of returns)
    if len(closing_prices) >= momentum_days:
        returns = np.diff(closing_prices[-(momentum_days + 1):]) / closing_prices[-(momentum_days + 1):-1]
        volatility = np.std(returns)
    else:
        volatility = 0
    features.append(volatility)

    # Feature 3: Volume Change (current volume vs average volume of last 5 days)
    avg_volume = np.mean(volumes[-min(5, len(volumes)):]) if len(volumes) >= 5 else 0
    current_volume = volumes[-1] if len(volumes) > 0 else 0
    volume_change = (current_volume - avg_volume) / (avg_volume if avg_volume > 0 else 1)  # Prevent division by zero
    features.append(volume_change)

    # Feature 4: Price Range (high - low over the last 20 days)
    price_range = np.max(high_prices[-20:]) - np.min(low_prices[-20:]) if len(high_prices) >= 20 else 0
    features.append(price_range)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY signals
        reward += 10.0 * features[2]  # Mild positive for SELL-aligned features (volume change)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Reward for aligning with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Positive reward for potential buy
        elif features[0] > 0:  # Overbought condition
            reward -= 5.0  # Penalty for buying in overbought conditions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))