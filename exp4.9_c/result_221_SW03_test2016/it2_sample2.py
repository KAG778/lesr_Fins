import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Price Change Rate (last day vs. the day before)
    price_change_rate = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 and closing_prices[-2] != 0 else 0.0

    # Feature 2: Volatility (using standard deviation of last 5 days of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0.0

    # Feature 3: Z-score of the last price against the 20-day mean
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:])
        z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0.0
    else:
        z_score = 0.0

    # Feature 4: Volume Change Rate (last volume vs. average of last 5 volumes)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1e-10  # prevent div by zero
    volume_change_rate = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0.0
    
    features = [price_change_rate, volatility, z_score, volume_change_rate]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate relative thresholds based on historical std
    historical_std = np.std(enhanced_s[123:])  # Use the historical std of features for dynamic thresholds
    risk_thresholds = {
        'low': 0.4 * historical_std,
        'medium': 0.7 * historical_std,
        'high': historical_std
    }

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_thresholds['high']:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10   # Mild positive reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_thresholds['medium']:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_thresholds['low']:
        z_score = enhanced_s[123]  # Assuming Z-Score is one of the features
        if z_score < -1:  # Oversold condition
            reward += 10  # Reward potential buy
        elif z_score > 1:  # Overbought condition
            reward += 10  # Reward potential sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std and risk_level < risk_thresholds['medium']:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds