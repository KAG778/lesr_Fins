import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and calculate daily returns
    closing_prices = s[0::6]  # Extract closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    
    # Feature 1: Exponential Moving Average (EMA)
    if len(closing_prices) >= 14:
        ema = np.mean(closing_prices[-14:])  # 14-day EMA
    else:
        ema = 0  # Handle edge case
    features.append(ema)

    # Feature 2: Bollinger Bands - Upper and Lower Bands
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
    else:
        upper_band = lower_band = 0  # Handle edge case
    features.append(upper_band)
    features.append(lower_band)

    # Feature 3: Average True Range (ATR)
    highs = s[2::6]
    lows = s[3::6]
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR
    features.append(atr)

    # Feature 4: Z-score of Daily Returns
    if len(daily_returns) > 0:
        mean_returns = np.mean(daily_returns)
        std_returns = np.std(daily_returns)
        z_score = (daily_returns[-1] - mean_returns) / std_returns if std_returns != 0 else 0
    else:
        z_score = 0  # Handle edge case
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Retrieve features
    reward = 0.0
    
    # Calculate historical thresholds for dynamic risk assessment
    historical_std = np.std(features)
    high_risk_threshold = historical_std * 1.5  # Using 1.5 standard deviations
    low_risk_threshold = historical_std * 0.5   # Using 0.5 standard deviations

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        reward -= 50  # Strong penalty for buying in high risk
        # Mild positive reward for SELL-aligned features
        reward += 10  # Mild reward for selling in high risk
    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        reward -= 20  # Moderate penalty for buying in moderate risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += 20  # Positive reward for upward trend
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += 20  # Positive reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < -1:  # Z-score indicating extreme oversold condition
            reward += 15  # Reward for buying in oversold conditions
        elif features[3] > 1:  # Z-score indicating extreme overbought condition
            reward -= 15  # Penalty for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is bounded
    return float(np.clip(reward, -100, 100))