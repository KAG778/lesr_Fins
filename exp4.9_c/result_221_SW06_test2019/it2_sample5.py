import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices, high, low, and volumes
    closing_prices = s[0:120:6]  # Closing prices
    high_prices = s[2:120:6]     # High prices
    low_prices = s[3:120:6]      # Low prices

    # Feature 1: Average True Range (ATR)
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                         abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    features.append(atr)

    # Feature 2: Z-score of the last closing price for mean-reversion
    mean_price = np.mean(closing_prices[-20:])  # Mean of last 20 closing prices
    std_price = np.std(closing_prices[-20:])    # Std dev of last 20 closing prices
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    features.append(z_score)

    # Feature 3: Rate of Change (ROC)
    roc_period = 10
    roc = (closing_prices[-1] - closing_prices[-(roc_period + 1)]) / closing_prices[-(roc_period + 1)] if len(closing_prices) > roc_period else 0
    features.append(roc)

    # Feature 4: Momentum divergence indicator
    momentum_divergence = (closing_prices[-1] - np.mean(closing_prices[-5:])) / np.std(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    features.append(momentum_divergence)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # The new features
    reward = 0.0

    # Calculate relative thresholds based on the historical standard deviation of the features
    historical_std = np.std(features)
    high_threshold = np.mean(features) + historical_std
    low_threshold = np.mean(features) - historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for high-risk BUY
        reward += np.random.uniform(5, 10)   # Mild positive for high-risk SELL

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[2] > 0:  # Uptrend with positive momentum
            reward += 20
        elif trend_direction < 0 and features[2] < 0:  # Downtrend with negative momentum
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < low_threshold:  # Oversold condition
            reward += 10  # Reward for buying oversold
        elif features[1] > high_threshold:  # Overbought condition
            reward -= 10  # Penalty for buying overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce magnitude of reward by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds