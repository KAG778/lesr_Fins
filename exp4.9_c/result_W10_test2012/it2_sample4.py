import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices

    # Feature 1: Exponential Moving Average (EMA) for trend detection
    def compute_ema(prices, period=20):
        if len(prices) < period:
            return np.mean(prices)
        return np.mean(prices[-period:])  # Simple EMA approximation

    ema = compute_ema(closing_prices)
    features.append(ema)

    # Feature 2: Average True Range (ATR) for volatility
    high_prices = s[2::6]
    low_prices = s[3::6]
    true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:], low_prices[1:] - closing_prices[:-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.mean(true_ranges)
    features.append(atr)

    # Feature 3: Rate of Change (ROC) for momentum
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) >= 15 and closing_prices[-15] != 0 else 0
    features.append(roc)

    # Feature 4: Bollinger Bands Width
    moving_avg = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    moving_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bb_width = moving_std / moving_avg if moving_avg != 0 else 0
    features.append(bb_width)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_risk_level = np.mean(enhanced_s[123])
    historical_risk_std = np.std(enhanced_s[123])

    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_risk_level + 1.5 * historical_risk_std:
        reward -= 50  # Strong negative for risky BUY-aligned features
        return max(-100, reward)  # Immediate return to prioritize risk management
    elif risk_level > historical_risk_level + 0.5 * historical_risk_std:
        reward -= 20  # Mild negative for BUY signals

    # Extract features
    features = enhanced_s[123:]

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < historical_risk_level:
        if trend_direction > 0 and features[2] > 0:  # Upward momentum
            reward += 30  # Strong positive for upward alignment
        elif trend_direction < 0 and features[2] < 0:  # Downward momentum
            reward += 30  # Strong positive for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < historical_risk_level:
        if features[3] < 30:  # Oversold condition
            reward += 20  # Reward for buying in an oversold market
        elif features[3] > 70:  # Overbought condition
            reward -= 20  # Penalize for buying in an overbought market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < historical_risk_level:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)