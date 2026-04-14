import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices

    # 1. Sharpe Ratio of the last N days returns
    N = 20
    returns = np.diff(closing_prices) / closing_prices[:-1]
    if len(returns) >= N:
        mean_return = np.mean(returns[-N:])
        std_return = np.std(returns[-N:])
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
    else:
        sharpe_ratio = 0
    features.append(sharpe_ratio)

    # 2. Bollinger Bands
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_avg + (std_dev * 2)
        lower_band = moving_avg - (std_dev * 2)
        features.append((closing_prices[-1] - lower_band) / (upper_band - lower_band))  # Bandwidth normalized
    else:
        features.append(0)

    # 3. Average True Range (ATR)
    if len(closing_prices) >= 14:
        high_prices = s[1::6]
        low_prices = s[2::6]
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    else:
        atr = 0
    features.append(atr)

    # 4. Z-Score of Momentum
    if len(returns) >= 20:
        momentum = returns[-1]
        momentum_mean = np.mean(returns[-20:])
        momentum_std = np.std(returns[-20:])
        z_score = (momentum - momentum_mean) / momentum_std if momentum_std > 0 else 0
        features.append(z_score)
    else:
        features.append(0)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical data (e.g., std deviation)
    mean_risk = 0.5  # Placeholder for historical mean risk level
    std_risk = 0.2   # Placeholder for historical std deviation of risk level
    risk_threshold_upper = mean_risk + 1.5 * std_risk
    risk_threshold_lower = mean_risk - 1.5 * std_risk

    # Priority 1: Risk Management
    if risk_level > risk_threshold_upper:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY actions
    elif risk_level > risk_threshold_lower:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY actions

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_lower:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Positive reward for BUY aligned features
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Positive reward for SELL aligned features

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_lower:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within [-100, 100]