import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    high_prices = s[2::6]      # Extracting high prices
    low_prices = s[3::6]       # Extracting low prices
    volume = s[4::6]           # Extracting trading volume

    # Feature 1: Exponential Moving Average (EMA) over the last 14 days
    ema = np.mean(closing_prices[-14:]) if len(closing_prices) >= 14 else 0

    # Feature 2: Rate of Change (ROC) over the last 14 days
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) >= 15 and closing_prices[-15] != 0 else 0

    # Feature 3: Chaikin Money Flow (CMF) - measuring the buying and selling pressure
    money_flow_volume = (closing_prices - low_prices) - (high_prices - closing_prices)
    cmf = np.sum(money_flow_volume[-14:] * volume[-14:]) / np.sum(volume[-14:]) if np.sum(volume[-14:]) != 0 else 0

    # Feature 4: Moving Average Convergence Divergence (MACD) - capturing trend shifts
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    macd = short_ema - long_ema

    features = [ema, roc, cmf, macd]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Determine thresholds based on historical standard deviation of risk levels
    historical_std = np.std([0.1, 0.4, 0.7])  # Using fixed values for simplicity; consider using historical data
    low_risk_threshold = 0.4 * historical_std
    high_risk_threshold = 0.7 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY signals in high risk
    elif risk_level > low_risk_threshold:
        reward += 20  # Mildly positive reward for SELL signals in moderate risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Reward for positive momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Reward for negative momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        if enhanced_s[123][2] < 30:  # Oversold
            reward += 15  # Reward for mean-reversion buy signal
        elif enhanced_s[123][2] > 70:  # Overbought
            reward += 15  # Reward for mean-reversion sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified bounds