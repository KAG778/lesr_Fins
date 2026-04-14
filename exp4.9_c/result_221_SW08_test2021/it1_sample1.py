import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    closing_prices = s[0:120:6]  # Extracting closing prices
    volumes = s[4:120:6]          # Extracting trading volumes

    # Feature 1: Exponential Moving Average (EMA) over the last 10 days
    ema = np.zeros(20)
    alpha = 2 / (10 + 1)  # Smoothing factor for EMA
    ema[0] = closing_prices[0]  # Starting point
    for i in range(1, len(closing_prices)):
        ema[i] = (closing_prices[i] * alpha) + (ema[i - 1] * (1 - alpha))

    # Feature 2: Price Volatility (standard deviation of the last 10 days)
    price_volatility = np.std(closing_prices[-10:]) if len(closing_prices) >= 10 else 0

    # Feature 3: Volume Spike (percentage change from average volume over the last 20 days)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    volume_spike = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    features = [ema[-1], price_volatility, volume_spike]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Utilize historical metrics to define thresholds
    historical_std = np.std(enhanced_s[123:])  # Use features to define dynamic thresholds
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std
    volatility_threshold = 0.6 * historical_std

    # Priority 1: Risk Management
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for BUY signals
        reward += 10  # Mild positive for SELL signals
    elif risk_level > 0.4 * historical_std:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold:  # Uptrend
            reward += 20  # Strong positive for bullish trend
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 20  # Strong positive for bearish trend

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += 10  # Reward mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > volatility_threshold and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is clamped to [-100, 100]
    return np.clip(reward, -100, 100)