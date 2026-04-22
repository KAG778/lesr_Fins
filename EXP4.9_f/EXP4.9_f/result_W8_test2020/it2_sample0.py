import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volume = s[4::6]          # Extract volume

    # Feature 1: Exponential Moving Average (EMA) for trend detection
    ema_short = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    ema_long = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    ema_diff = ema_short - ema_long  # Distance between short and long EMA
    
    # Feature 2: Rate of Change (ROC) for momentum
    roc = ((closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]) * 100 if len(closing_prices) >= 6 else 0

    # Feature 3: Stochastic Oscillator (K %)
    lowest_low = np.min(low_prices[-14:]) if len(low_prices) >= 14 else 0
    highest_high = np.max(high_prices[-14:]) if len(high_prices) >= 14 else 1  # Avoid division by zero
    stoch_k = ((closing_prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100 if highest_high - lowest_low != 0 else 0

    # Feature 4: On-Balance Volume (OBV)
    obv = np.sum(np.where(closing_prices[1:] > closing_prices[:-1], volume[1:], -volume[1:])) if len(closing_prices) > 1 else 0

    features = [ema_diff, roc, stoch_k, obv]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Get historical standard deviation from previous risk levels for dynamic thresholds
    risk_std = np.std(enhanced_s[120:123])  # Assuming the regime vector is dynamically updated
    high_risk_threshold = 0.7 * risk_std
    medium_risk_threshold = 0.4 * risk_std

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY signals in high risk
    elif risk_level > medium_risk_threshold:
        reward += 20  # Mildly positive reward for SELL signals in moderate risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Reward for positive momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Reward for negative momentum (shorting)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        if enhanced_s[123][2] < 30:  # Oversold
            reward += 15  # Reward for mean-reversion buy signal
        elif enhanced_s[123][2] > 70:  # Overbought
            reward += 15  # Reward for mean-reversion sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within specified bounds