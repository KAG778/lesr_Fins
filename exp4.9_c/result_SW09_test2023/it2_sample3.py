import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: 14-day Relative Strength Index (RSI)
    if len(closing_prices) >= 14:
        price_changes = np.diff(closing_prices[-14:])
        gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
        loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI when insufficient data
    features.append(rsi)

    # Feature 2: Price Momentum (current closing price - closing price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0
    features.append(price_momentum)

    # Feature 3: 20-day Moving Average
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    features.append(moving_average)

    # Feature 4: Crisis Signal (1 if current price is below 95% of the moving average, else 0)
    crisis_signal = 1 if closing_prices[-1] < 0.95 * moving_average else 0
    features.append(crisis_signal)

    # Feature 5: Recent Volume Spike Indicator (current volume vs. average volume over the last 14 days)
    avg_volume = np.mean(volumes[-14:]) if len(volumes) >= 14 else volumes[-1]
    volume_spike = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_spike)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Historical thresholds (these should be dynamically calculated based on historical data)
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    trend_threshold = 0.3

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative for BUY-aligned features
        reward += 10   # Mild positive for SELL-aligned features (risk-off)
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Strong reward for alignment with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(max(-100, min(100, reward)))