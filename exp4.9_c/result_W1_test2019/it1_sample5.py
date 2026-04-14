import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    days = len(closing_prices)

    # Feature 1: Historical Volatility (30-day rolling std)
    historical_volatility = np.std(closing_prices[-30:]) if days >= 30 else 0

    # Feature 2: 20-day Moving Average
    moving_average_20 = np.mean(closing_prices[-20:]) if days >= 20 else 0

    # Feature 3: 50-day Moving Average
    moving_average_50 = np.mean(closing_prices[-50:]) if days >= 50 else 0

    # Feature 4: Price Change from Moving Average (20-day)
    price_change_from_ma = (closing_prices[-1] - moving_average_20) / moving_average_20 if moving_average_20 != 0 else 0

    # Collect features
    features = [historical_volatility, price_change_from_ma]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical mean and std for relative thresholds
    # Assuming we have historical data to compute these
    historical_mean_risk = 0.5  # Placeholder: Replace with actual historical mean risk
    historical_std_risk = 0.2    # Placeholder: Replace with actual historical std of risk
    
    # Define relative thresholds
    high_risk_threshold = historical_mean_risk + 1.5 * historical_std_risk
    low_risk_threshold = historical_mean_risk - 1.5 * historical_std_risk

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -50  # STRONG NEGATIVE reward for BUY-aligned features
        reward += 20   # Mild positive reward for SELL
        return np.clip(reward, -100, 100)  # Early exit due to high risk
    elif risk_level > low_risk_threshold:
        reward += -10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:
            reward += 25  # Reward for positive trend
        else:
            reward += 15  # Reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)