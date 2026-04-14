import numpy as np

def revise_state(s):
    # s: 120d raw state (OHLCV data)
    closing_prices = s[0::6]  # Extracting closing prices
    opening_prices = s[1::6]  # Extracting opening prices
    high_prices = s[2::6]     # Extracting high prices
    low_prices = s[3::6]      # Extracting low prices
    volumes = s[4::6]         # Extracting volumes

    # Feature 1: Exponential Moving Average (EMA) over the last 20 days
    ema = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0

    # Feature 2: Average True Range (ATR) for volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], closing_prices[1:-1] - closing_prices[2:])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0  # 14-day ATR

    # Feature 3: Momentum (Rate of Change) over the last 5 days
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0.0

    features = [ema, atr, momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY in high risk
        reward += 10.0 if features[2] < 0 else 0  # Positive for SELL if momentum is negative
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY in elevated risk

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[2] * 20.0  # Reward for aligning with momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        if features[2] < 0:  # Oversold condition
            reward += 15.0  # Reward for potential buy signal
        elif features[2] > 0:  # Overbought condition
            reward += -10.0  # Penalize for potential sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in uncertain markets

    return float(np.clip(reward, -100, 100))