import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices and volumes
    closing_prices = s[0::6]  # Closing prices of the last 20 days
    volumes = s[4::6]          # Trading volumes of the last 20 days
    high_prices = s[2::6]      # High prices of the last 20 days
    low_prices = s[3::6]       # Low prices of the last 20 days
    
    # Feature 1: Average True Range (ATR) - captures volatility
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR
    features.append(atr)

    # Feature 2: Rate of Change (ROC) - captures momentum
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) > 15 else 0
    features.append(roc)

    # Feature 3: Volume Average - normalized volume against the average
    avg_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 1  # Avoid division by zero
    current_volume = volumes[-1]
    volume_avg_feature = (current_volume - avg_volume) / avg_volume  # Normalize change in volume
    features.append(volume_avg_feature)

    # Feature 4: 14-day RSI - captures overbought/oversold conditions
    window_length = 14
    rsi = 50  # Default to neutral
    if len(closing_prices) >= window_length:
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
        avg_gain = np.mean(gains[-window_length:])
        avg_loss = np.mean(losses[-window_length:])
        rs = avg_gain / avg_loss if avg_loss > 0 else np.inf
        rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate dynamic thresholds based on historical volatility
    historical_volatility = np.std(enhanced_s[123:])  # Use features from revise_state for volatility reference
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.4 * historical_volatility

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for buying in high risk
        reward += np.random.uniform(5, 10)    # Mild positive for selling in high risk
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(5, 15)    # Moderate negative for buy signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 20  # Positive reward for bullish features if trend is up
        if trend_direction < -0.3:  # If trend is down
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_volatility and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Constrain reward within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward