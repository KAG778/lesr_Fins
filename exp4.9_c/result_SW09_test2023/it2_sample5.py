import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: 14-day Relative Strength Index (RSI) for trend detection
    if len(closing_prices) >= 14:
        price_changes = np.diff(closing_prices[-14:])
        gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
        loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI for insufficient data
    features.append(rsi)

    # Feature 2: 14-day Moving Average
    moving_average = np.mean(closing_prices[-14:]) if len(closing_prices) >= 14 else closing_prices[-1]
    features.append(moving_average)

    # Feature 3: Price Momentum (current closing price vs price 3 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-4] if len(closing_prices) >= 4 else 0
    features.append(price_momentum)

    # Feature 4: Maximum Drawdown over the last 20 days
    peak = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    drawdown = (peak - closing_prices[-1]) / peak if peak != 0 else 0
    features.append(drawdown)

    # Feature 5: Recent Volume Spike Indicator (current volume vs. average volume)
    avg_volume = np.mean(volumes[-14:]) if len(volumes) >= 14 else 0
    volume_spike = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_spike)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for relative risk assessment
    historical_std = np.std(enhanced_s[:120])  # Use the standard deviation of the last 120 observations
    risk_threshold_high = 0.5 * historical_std
    risk_threshold_medium = 0.25 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -40  # Strong negative for BUY-aligned features
        reward += 5     # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Reward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))

    return reward