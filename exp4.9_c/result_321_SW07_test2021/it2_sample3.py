import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Exponential Moving Average Convergence Divergence (MACD)
    closing_prices = s[0::6]  # Extracting closing prices
    ema_short = np.mean(closing_prices[-12:]) if len(closing_prices[-12:]) > 0 else 0
    ema_long = np.mean(closing_prices[-26:]) if len(closing_prices[-26:]) > 0 else 0
    macd = ema_short - ema_long
    features.append(macd)

    # Feature 2: Price Momentum (current close - close 5 days ago)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(price_momentum)

    # Feature 3: Average True Range (ATR) for volatility measure
    highs = s[2::6]
    lows = s[3::6]
    tr = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[:-1])
    tr = np.maximum(tr, closing_prices[:-1] - lows[1:])
    atr = np.mean(tr[-14:]) if len(tr) > 0 else 0
    features.append(atr)

    # Feature 4: Z-Score of the last 5 closing prices to identify overbought/oversold conditions
    mean_price = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    std_price = np.std(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 1  # Avoid division by zero
    z_score = (closing_prices[-1] - mean_price) / std_price
    features.append(z_score)

    # Feature 5: Volume Change Rate (current volume versus average volume over the last 5 days)
    current_volume = s[4::6][-1] if len(s[4::6]) > 0 else 0
    avg_volume = np.mean(s[4::6][-5:]) if len(s[4::6][-5:]) > 0 else 1  # Avoid division by zero
    volume_change_rate = (current_volume - avg_volume) / avg_volume
    features.append(volume_change_rate)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Calculate historical standard deviation for dynamic thresholds
    historical_std = np.std(features)
    relative_risk_threshold_high = 0.7 * historical_std
    relative_risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > relative_risk_threshold_high:
        reward += -50  # Strong negative for BUY-aligned features
        if features[0] < 0:  # If MACD indicates a downtrend, consider it a sell scenario
            reward += 10  # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)
    
    elif risk_level > relative_risk_threshold_medium:
        reward += -20  # Moderate negative for BUY signals
        return np.clip(reward, -100, 100)
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < relative_risk_threshold_medium:
        if (trend_direction > 0 and features[1] > 0) or (trend_direction < 0 and features[1] < 0):
            reward += 20  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < relative_risk_threshold_medium:
        if features[3] < -1:  # Assuming z-score < -1 indicates oversold
            reward += 10  # Reward for buying in oversold condition
        elif features[3] > 1:  # Assuming z-score > 1 indicates overbought
            reward += 10  # Reward for selling in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the bounds