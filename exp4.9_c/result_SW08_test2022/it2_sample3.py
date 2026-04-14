import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]          # Trading volumes

    # Feature 1: Exponential Moving Average (EMA) to capture trends more dynamically
    ema_short = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0
    ema_long = np.mean(closing_prices[-30:]) if len(closing_prices) >= 30 else 0
    features.append(ema_short - ema_long)  # EMA difference

    # Feature 2: Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # High prices
    low_prices = s[3:120:6]   # Low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR
    features.append(atr)

    # Feature 3: Rate of Change (ROC) to capture momentum
    roc_period = 10
    if len(closing_prices) >= roc_period:
        roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period]
    else:
        roc = 0
    features.append(roc)

    # Feature 4: Volume Weighted Average Price (VWAP) for trading volume analysis
    total_volume = np.sum(volumes[-20:]) if len(volumes) >= 20 else 0
    vwap = np.sum(closing_prices[-20:] * volumes[-20:]) / total_volume if total_volume != 0 else 0
    features.append(vwap)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_returns = enhanced_s[123:]  # Assuming features start at index 123
    mean_return = np.mean(historical_returns)
    std_return = np.std(historical_returns)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > (mean_return + 2 * std_return):  # Using dynamic threshold
        reward -= 50  # Strong negative reward for BUY-aligned features (high risk)
        reward += 10 if enhanced_s[123] < 0 else 0  # Mild positive reward for SELL-aligned features
    elif risk_level > (mean_return + std_return):
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < (mean_return + std_return):
        reward += 15 if trend_direction > 0 else -15  # Positive for uptrend, negative for downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < (mean_return):
        reward += 10 if enhanced_s[123] < 0 else 0  # Reward mean-reversion features if oversold
        reward -= 5 if enhanced_s[123] > 0 else 0  # Penalize breakout-chasing features if overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < (mean_return + std_return):
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds