import numpy as np

def revise_state(s):
    # Extract relevant price and volume data
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: Exponential Moving Average (EMA) for trend detection
    ema_short = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    ema_long = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    ema_trend = ema_short - ema_long  # EMA difference for trend strength

    # Feature 2: Rate of Change (ROC) to measure momentum
    if len(closing_prices) > 5:
        roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]  # 5-day ROC
    else:
        roc = 0.0

    # Feature 3: Volatility based on historical standard deviation
    historical_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0  # 20-day volatility

    # Feature 4: Volume Weighted Average Price (VWAP)
    if len(volumes) >= 20:
        vwap = np.sum(closing_prices[-20:] * volumes[-20:]) / np.sum(volumes[-20:])
    else:
        vwap = 0.0

    features = [ema_trend, roc, historical_volatility, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Determine relative thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])
    high_risk_threshold = historical_std * 1.5  # Dynamic threshold for high risk
    trend_threshold = 0.3 * historical_std  # Dynamic threshold for trend detection

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for risky BUY-aligned features
        reward += 10   # Mild positive for SELL-aligned features if risk is high
        return np.clip(reward, -100, 100)  # Early return if in high-risk environment

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level <= 0.4:
        momentum_feature = enhanced_s[123]  # This would be the momentum feature from revise_state
        if trend_direction > 0:
            reward += 20 if momentum_feature > 0 else -10  # Positive reward for alignment
        else:
            reward += 20 if momentum_feature < 0 else -10  # Positive reward for alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]