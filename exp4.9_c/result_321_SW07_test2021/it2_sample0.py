import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Exponential Moving Average (EMA) - Short-term (5 days)
    closing_prices = s[0::6]  # Extracting closing prices
    ema_short = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    features.append(ema_short)
    
    # Feature 2: Price Rate of Change (ROC)
    price_roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(price_roc)

    # Feature 3: Average True Range (ATR) - Volatility measure
    highs = s[2::6]
    lows = s[3::6]
    tr = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    features.append(atr)

    # Feature 4: Z-score of the last 5 closing prices
    mean_price = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    std_price = np.std(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 1  # Avoid division by zero
    z_score = (closing_prices[-1] - mean_price) / std_price
    features.append(z_score)

    # Feature 5: Standard Deviation of Daily Returns (Volatility)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if len(returns) > 0 else 0
    features.append(volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate historical std dev for relative thresholds
    historical_std = np.std(features) if features.size > 0 else 1
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -50  # Strong negative reward for BUY-aligned features
        if features[0] < 0:  # If EMA short is below EMA long (aligned with selling)
            reward += 10  # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)

    elif risk_level > risk_threshold_medium:
        reward += -20  # Moderate negative reward for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if (trend_direction > 0 and features[1] > 0) or (trend_direction < 0 and features[1] < 0):
            reward += 20  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < risk_threshold_medium:
        if features[3] < -1:  # Assuming z-score < -1 indicates oversold
            reward += 10  # Reward for buying in oversold condition
        elif features[3] > 1:  # Assuming z-score > 1 indicates overbought
            reward += 10  # Reward for selling in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds