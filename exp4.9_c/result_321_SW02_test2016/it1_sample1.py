import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    # Feature 1: 5-day Price Momentum
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Average Volume Change (relative to historical average)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
    current_volume = volumes[-1] if len(volumes) > 0 else 0
    volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0

    # Feature 3: Price Divergence from 20-day Moving Average
    ma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    price_divergence = (closing_prices[-1] - ma_20) / ma_20 if ma_20 != 0 else 0

    # Feature 4: Historical Volatility (20-day rolling standard deviation)
    if len(closing_prices) > 20:
        returns = np.diff(closing_prices[-20:]) / closing_prices[-21:-1]
        historical_volatility = np.std(returns)
    else:
        historical_volatility = 0

    # Compile features into a single array
    features = [price_momentum, volume_change, price_divergence, historical_volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical data (e.g., standard deviations)
    risk_threshold = 0.7  # This can be adjusted based on historical risk levels
    trend_threshold = 0.3  # Same for trend sensitivity

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative penalty for buying in high risk
        reward += 20 * (1 - risk_level)  # Mild positive for selling
    elif risk_level > 0.4:
        reward -= 20  # Moderate penalty for buying in elevated risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold:
            reward += 25  # Strong positive for bullish alignment
        elif trend_direction < -trend_threshold:
            reward += 25  # Strong positive for bearish alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    # Ensure that the reward is within the bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return float(reward)