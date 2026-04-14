import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Extract closing prices
    num_prices = len(closing_prices)

    # Feature 1: Relative Momentum (normalized momentum)
    if num_prices > 5:
        momentum = (closing_prices[-1] - closing_prices[-6]) / np.std(closing_prices[-6:]) if np.std(closing_prices[-6:]) != 0 else 0
        features.append(momentum)
    else:
        features.append(0)

    # Feature 2: Volume Change (normalized)
    volumes = s[4::6]  # Extract trading volumes
    if len(volumes) > 1 and volumes[-2] > 0:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
        features.append(volume_change)
    else:
        features.append(0)

    # Feature 3: Average True Range (ATR) for volatility measure
    highs = s[2::6]
    lows = s[3::6]
    true_ranges = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    features.append(atr)

    # Feature 4: Z-score of daily returns for mean reversion signal
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    if len(daily_returns) > 0:
        mean_returns = np.mean(daily_returns)
        std_returns = np.std(daily_returns)
        z_score = (daily_returns[-1] - mean_returns) / std_returns if std_returns != 0 else 0
    else:
        z_score = 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate dynamic thresholds for risk evaluation based on features
    vol_threshold = np.mean(features[2]) + 2 * np.std(features[2])  # ATR as volatility measure
    momentum_threshold = np.mean(features[0])  # Momentum measure

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > momentum_threshold:  # Positive momentum indicates a BUY signal
            reward -= np.random.uniform(40, 60)  # Strong negative for BUY
        else:
            reward += np.random.uniform(5, 15)  # Mild positive for SELL

    elif risk_level > 0.4:
        if features[0] > momentum_threshold:  # Positive momentum
            reward -= np.random.uniform(20, 30)  # Moderate negative for BUY

    # Priority 2 — TREND FOLLOWING (only when risk is low)
    if risk_level < 0.4:
        if abs(trend_direction) > 0.3:
            if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
                reward += np.random.uniform(10, 20)  # Reward for alignment
            elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
                reward += np.random.uniform(10, 20)  # Reward for alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < -1:  # Z-score indicating extreme oversold condition
            reward += np.random.uniform(5, 15)  # Reward for buying in oversold conditions
        elif features[3] > 1:  # Z-score indicating extreme overbought condition
            reward -= np.random.uniform(5, 15)  # Penalty for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within bounds