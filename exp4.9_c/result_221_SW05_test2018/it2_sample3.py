import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state.

    s: 120d raw state
    Returns ONLY new features (NOT including s or regime).
    """
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Exponential Moving Average (EMA) over the last 10 days
    ema_period = 10
    if len(closing_prices) >= ema_period:
        weights = np.exp(np.linspace(-1, 0, ema_period))
        weights /= weights.sum()
        ema = np.convolve(closing_prices, weights, mode='valid')[-1]
    else:
        ema = closing_prices[-1] if len(closing_prices) > 0 else 0
    
    # Feature 2: Price momentum (current price vs. moving average of the last 5 days)
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    price_momentum = (closing_prices[-1] - moving_average) / (moving_average if moving_average != 0 else 1)

    # Feature 3: Average True Range (ATR) for volatility measure
    true_ranges = np.maximum(high_prices[-5:] - low_prices[-5:], np.abs(closing_prices[-5:] - opening_prices[-5:]))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0

    # Feature 4: Current price relative to the 20-day moving average
    ma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else np.mean(closing_prices)
    price_relative_to_ma = (closing_prices[-1] - ma_20) / (ma_20 if ma_20 != 0 else 1)

    # Feature 5: Volume Change (current volume vs. average volume of the last 20 days)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
    volume_change = (volumes[-1] - avg_volume) / (avg_volume if avg_volume != 0 else 1)

    return np.array([price_momentum, ema, atr, price_relative_to_ma, volume_change])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    price_momentum = features[0]
    atr = features[2]
    price_relative_to_ma = features[3]

    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_std = np.std(features)  # Use historical std of features for dynamic thresholds
    low_risk_threshold = 0.4 * historical_std
    high_risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if price_momentum > 0:  # Expectation of upward momentum
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        elif price_momentum < 0:  # Expectation of downward momentum
            reward += np.random.uniform(5, 10)

    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        if price_momentum > 0:
            reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level < low_risk_threshold:
        if trend_direction > trend_threshold and price_momentum > 0:
            reward += np.random.uniform(10, 20)  # Align with upward trend
        elif trend_direction < -trend_threshold and price_momentum < 0:
            reward += np.random.uniform(10, 20)  # Align with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if price_relative_to_ma < -0.1:  # Oversold condition
            reward += np.random.uniform(10, 20)
        elif price_relative_to_ma > 0.1:  # Overbought condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]