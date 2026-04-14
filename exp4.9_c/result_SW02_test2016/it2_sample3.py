import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract volumes

    if len(closing_prices) < 20 or len(volumes) < 20:
        return np.zeros(4)  # Return zeros if there are not enough data points

    # Feature 1: Historical Volatility (Standard Deviation of closing prices over 20 days)
    historical_volatility = np.std(closing_prices[-20:])

    # Feature 2: Price Distance to 20-day Moving Average
    sma_20 = np.mean(closing_prices[-20:])
    price_distance_to_sma = (closing_prices[-1] - sma_20) / historical_volatility if historical_volatility != 0 else 0

    # Feature 3: Rate of Change (ROC) over the last 14 days
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if closing_prices[-15] != 0 else 0

    # Feature 4: Momentum Oscillator (difference between 10-day and 20-day moving averages)
    ma_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    ma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    momentum_oscillator = ma_10 - ma_20

    features = [historical_volatility, price_distance_to_sma, roc, momentum_oscillator]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical risk based on features
    historical_risk = np.std(enhanced_s[123:])  # Assuming features are in the context of risk
    risk_threshold_high = 0.7 * historical_risk  # High risk threshold
    risk_threshold_moderate = 0.4 * historical_risk  # Moderate risk threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += 30  # Positive reward for buying in an uptrend
        else:  # Downtrend
            reward += 30  # Positive reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_threshold_moderate:
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range