import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Exponential Moving Average (EMA) over the last 14 days
    ema_short = np.mean(closing_prices[-14:]) if len(closing_prices) >= 14 else 0
    ema_long = np.mean(closing_prices[-50:]) if len(closing_prices) >= 50 else 0
    ema_feature = ema_short - ema_long  # Difference between short and long EMA

    # Feature 2: MACD
    macd_line = ema_short - ema_long  # MACD line
    signal_line = np.mean(closing_prices[-9:]) if len(closing_prices) >= 9 else 0  # Signal line
    macd_feature = macd_line - signal_line

    # Feature 3: Stochastic Oscillator
    lowest_low = np.min(low_prices[-14:]) if len(low_prices) >= 14 else 0
    highest_high = np.max(high_prices[-14:]) if len(high_prices) >= 14 else 0
    stoch_feature = ((closing_prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100 if highest_high != lowest_low else 0

    # Feature 4: Volume Change (percentage change over the last 5 days)
    volume_change = (volumes[-1] - volumes[-6]) / volumes[-6] if len(volumes) >= 6 and volumes[-6] != 0 else 0

    features = [ema_feature, macd_feature, stoch_feature, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate historical thresholds for risk levels
    risk_std = np.std([0.1, 0.4, 0.7])  # Placeholder for dynamic calculation
    high_risk_threshold = 0.7 * risk_std
    low_risk_threshold = 0.4 * risk_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY signals in high risk
    elif risk_level > low_risk_threshold:
        reward += 20  # Mildly positive reward for SELL signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Reward for positive momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Reward for negative momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        if enhanced_s[123][2] < 30:  # Oversold
            reward += 15  # Buy signal
        elif enhanced_s[123][2] > 70:  # Overbought
            reward += 15  # Sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clip reward to be within [-100, 100]
    return float(np.clip(reward, -100, 100))