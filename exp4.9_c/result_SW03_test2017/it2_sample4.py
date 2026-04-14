import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0:120:6]  # Every 6th element starting from index 0 is the closing price
    volumes = s[4:120:6]         # Extract trading volumes
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices

    # Feature 1: Rate of Change (Price Momentum)
    if len(closing_prices) > 1 and closing_prices[-2] != 0:
        momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        momentum = 0.0
    features.append(momentum)

    # Feature 2: Average Volume Change (relative to historical average)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes) if len(volumes) > 0 else 0
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_change)

    # Feature 3: Historical Volatility (20-day)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns) * np.sqrt(20) if len(returns) > 0 else 0
    features.append(historical_volatility)

    # Feature 4: Bollinger Bands % (relative position)
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        if std_dev > 0:
            bb_percent = (closing_prices[-1] - (mean_price - 2 * std_dev)) / (4 * std_dev)  # Normalize
        else:
            bb_percent = 0.0
    else:
        bb_percent = 0.0
    features.append(bb_percent)

    # Feature 5: Z-score of current price relative to historical mean
    mean_price = np.mean(closing_prices[-20:])  # Last 20 days
    std_dev = np.std(closing_prices[-20:]) if np.std(closing_prices[-20:]) > 0 else 1  # Avoid division by zero
    z_score = (closing_prices[-1] - mean_price) / std_dev
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Using standard deviation of features
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 15)    # Mild positive for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 25 * abs(trend_direction)  # Reward momentum alignment based on strength of trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]