import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Adjusted Momentum (momentum normalized by historical volatility)
    if len(closing_prices) > 6:
        momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]  # Current vs. 5 days ago
        historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1])  # Historical volatility
        adjusted_momentum = momentum / (historical_volatility + 1e-9)  # Avoid division by zero
    else:
        adjusted_momentum = 0
    features.append(adjusted_momentum)

    # Feature 2: Relative Volume (compared to the 20-day average)
    if len(volumes) > 20:
        avg_volume = np.mean(volumes[-20:])  # 20-day average volume
        relative_volume = volumes[-1] / avg_volume if avg_volume > 0 else 0
    else:
        relative_volume = 0
    features.append(relative_volume)

    # Feature 3: Z-score of Daily Returns for volatility measurement
    if len(closing_prices) > 1:
        daily_returns = np.diff(closing_prices) / closing_prices[:-1]
        mean_returns = np.mean(daily_returns)
        std_returns = np.std(daily_returns)
        z_score = (daily_returns[-1] - mean_returns) / std_returns if std_returns != 0 else 0
    else:
        z_score = 0
    features.append(z_score)

    # Feature 4: Average True Range (ATR) for volatility assessment
    highs = s[2::6]
    lows = s[3::6]
    if len(highs) > 1 and len(lows) > 1:
        true_ranges = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
        atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    else:
        atr = 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract the features
    reward = 0.0

    # Calculate relative thresholds for risk management
    historical_std = np.std(features)
    high_risk_threshold = historical_std * 1.5
    low_risk_threshold = historical_std * 0.5

    # Priority 1: Risk Management
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Adjusted momentum indicates a BUY signal
            reward -= np.random.uniform(40, 60)  # Strong penalty for buying in high risk
        else:
            reward += np.random.uniform(10, 20)  # Mild positive reward for SELL-aligned features
    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Adjusted momentum
            reward -= np.random.uniform(20, 30)

    # Priority 2: Trend Following (only when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += np.random.uniform(15, 30)
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += np.random.uniform(15, 30)

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Z-score indicating extreme oversold condition
            reward += np.random.uniform(5, 15)  # Reward for buying in oversold conditions
        elif features[2] > 1:  # Z-score indicating extreme overbought condition
            reward -= np.random.uniform(5, 15)  # Penalty for selling in overbought conditions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds