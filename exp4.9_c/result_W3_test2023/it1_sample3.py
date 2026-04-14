import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Price Momentum (current closing price vs closing price 5 days ago)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 6 and closing_prices[-6] != 0 else 0

    # Feature 2: Average Volume (last 10 days)
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0

    # Feature 3: Z-Score of Closing Prices (indicates how far prices are from the mean)
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:])
        z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    else:
        z_score = 0

    # Feature 4: Average True Range (ATR) for volatility measurement (last 14 days)
    if len(closing_prices) >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:]) - np.minimum(low_prices[1:], closing_prices[:-1])
        atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    else:
        atr = 0

    features = [price_momentum, avg_volume, z_score, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative for BUY-aligned features
        reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
    elif risk_level > 0.4:
        # Mild negative for BUY signals
        reward = -10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive reward for upward momentum
        else:  # Downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Oversold situation
            reward += 15  # Reward for buying in mean-reversion
        else:  # Overbought situation
            reward -= 15  # Penalize for selling in mean-reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the range [-100, 100]
    return float(np.clip(reward, -100, 100))