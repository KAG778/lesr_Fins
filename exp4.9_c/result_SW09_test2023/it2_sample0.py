import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: 14-day Relative Strength Index (RSI) for trend detection
    rsi_period = 14
    if len(closing_prices) >= rsi_period:
        price_changes = np.diff(closing_prices[-rsi_period:])
        gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
        loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI when insufficient data
    features.append(rsi)

    # Feature 2: 5-day Moving Average (MA) for trend confirmation
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    features.append(moving_average)

    # Feature 3: Volatility Measure (Standard Deviation of Closing Prices over last 5 days)
    volatility = np.std(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    features.append(volatility)

    # Feature 4: Crisis Indicator (Percentage drop from the peak in the last 20 days)
    peak_price = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    crisis_indicator = (peak_price - closing_prices[-1]) / peak_price if peak_price > 0 else 0
    features.append(crisis_indicator)

    # Feature 5: Volume Change Percentage (current volume vs average volume over the last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
    volume_change_percentage = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_change_percentage)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for risk level (using historical volatility)
    historical_std = np.std(enhanced_s[0:120])  # Using previous 120 data points for std
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative for BUY-aligned features
        reward += 5    # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += 15  # Reward for upward momentum
        else:  # Downtrend
            reward += 15  # Reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))

    return reward