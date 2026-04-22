import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Exponential Moving Average (EMA) - 10 days
    if len(closing_prices) >= 10:
        weights = np.arange(1, 11)
        ema_10 = np.sum(closing_prices[-10:] * weights) / np.sum(weights)
    else:
        ema_10 = 0.0

    # Feature 2: Sharpe Ratio of returns over the last 10 days
    if len(closing_prices) >= 10:
        returns = np.diff(closing_prices[-10:]) / closing_prices[-10:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Feature 3: Average True Range (ATR) - Measure of volatility
    if len(closing_prices) >= 2:
        tr = np.maximum(closing_prices[1:] - closing_prices[:-1], 
                        np.maximum(closing_prices[1:] - closing_prices[:-1], 
                                   closing_prices[:-1] - closing_prices[1:]))
        atr = np.mean(tr[-10:]) if len(tr) >= 10 else 0.0
    else:
        atr = 0.0

    features = [ema_10, sharpe_ratio, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    avg_risk = np.mean(features)  # Mean of features as a proxy for risk
    std_risk = np.std(features)    # Standard deviation for risk thresholding
    risk_threshold_high = avg_risk + 2 * std_risk
    risk_threshold_mid = avg_risk + std_risk

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50) if features[0] > 0 else 0  # Positive EMA indicates a buy signal
        # Mild positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10) if features[0] < 0 else 0  # Negative EMA indicates a sell signal
    elif risk_level > risk_threshold_mid:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20) if features[0] > 0 else 0

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_mid:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20) if features[0] > 0 else 0  # Reward if EMA aligns with trend
        elif trend_direction < 0:  # Downtrend
            reward += np.random.uniform(10, 20) if features[0] < 0 else 0  # Reward if EMA aligns with trend

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < -0.05:  # Strongly oversold condition based on Sharpe ratio
            reward += np.random.uniform(10, 20)  # Reward for buying
        elif features[1] > 0.05:  # Strongly overbought condition based on Sharpe ratio
            reward += np.random.uniform(10, 20)  # Reward for selling

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]