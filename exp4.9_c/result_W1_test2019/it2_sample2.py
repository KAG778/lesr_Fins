import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element)
    days = len(closing_prices)

    # Feature 1: Price Change from Moving Average (50-day)
    moving_average_50 = np.mean(closing_prices[-50:]) if days >= 50 else 0
    price_change_from_ma = (closing_prices[-1] - moving_average_50) / moving_average_50 if moving_average_50 != 0 else 0

    # Feature 2: Rate of Change (ROC) - Momentum
    roc = (closing_prices[-1] - closing_prices[-21]) / closing_prices[-21] if days >= 21 and closing_prices[-21] != 0 else 0

    # Feature 3: Historical Volatility (30-day rolling std)
    historical_volatility = np.std(closing_prices[-30:]) if days >= 30 else 0

    # Feature 4: Volume Oscillator (difference between short-term and long-term EMA of volume)
    volumes = s[4::6]  # Extract volumes
    ema_short_volume = np.mean(volumes[-5:]) if days >= 5 else 0
    ema_long_volume = np.mean(volumes[-20:]) if days >= 20 else 0
    volume_oscillator = ema_short_volume - ema_long_volume

    # Collect features
    features = [price_change_from_ma, roc, historical_volatility, volume_oscillator]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for relative thresholds
    historical_mean_risk = 0.5  # Placeholder: Replace with actual historical mean risk
    historical_std_risk = 0.2     # Placeholder: Replace with actual historical std of risk
    high_risk_threshold = historical_mean_risk + historical_std_risk
    low_risk_threshold = historical_mean_risk - historical_std_risk

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
        reward += 20   # Mild positive for SELL
        return np.clip(reward, -100, 100)  # Early exit due to high risk
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 20 * np.sign(trend_direction)  # Reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)