import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Exponential Moving Average (EMA) - 5 days
    closing_prices = s[0::6]  # Closing prices
    ema_5 = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    features.append(ema_5)

    # Feature 2: Price Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(roc)

    # Feature 3: Average True Range (ATR) - Volatility measure
    highs = s[2::6]
    lows = s[3::6]
    tr = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # Average True Range
    features.append(atr)

    # Feature 4: Z-score of the last 5 closing prices
    mean_price = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    std_price = np.std(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 1  # Avoid division by zero
    z_score = (closing_prices[-1] - mean_price) / std_price
    features.append(z_score)

    # Feature 5: Momentum Indicator - Difference between current price and price 5 days ago
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0
    features.append(price_momentum)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate dynamic thresholds based on historical statistics
    historical_std = np.std(features) if features.size > 0 else 1
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -50  # Strong negative for BUY-aligned features
        reward += 10 if features[-1] < 0 else -10  # Mild positive for SELL-aligned features based on momentum
        return np.clip(reward, -100, 100)
    
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if (trend_direction > 0 and features[1] > 0) or (trend_direction < 0 and features[1] < 0):
            reward += 20  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < -1:  # Assuming z-score < -1 indicates oversold
            reward += 10  # Reward for buying in oversold condition
        elif features[3] > 1:  # Assuming z-score > 1 indicates overbought
            reward += 10  # Reward for selling in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the bounds