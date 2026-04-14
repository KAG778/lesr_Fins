import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]
    volumes = s[4:120:6]

    # Feature 1: Relative Strength Index (RSI) over the last 14 days
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50.0  # Neutral RSI when not enough data
    features.append(rsi)

    # Feature 2: Price Momentum (10-day)
    price_momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) >= 11 else 0
    features.append(price_momentum)

    # Feature 3: Average True Range (ATR) (14-day)
    high_prices = s[1:120:6]  # High prices for 20 days
    low_prices = s[2:120:6]   # Low prices for 20 days
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    features.append(atr)

    # Feature 4: Price Divergence from 50-day Moving Average
    ma_50 = np.mean(closing_prices[-50:]) if len(closing_prices) >= 50 else closing_prices[-1]
    price_divergence = (closing_prices[-1] - ma_50) / ma_50 if ma_50 != 0 else 0
    features.append(price_divergence)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds based on past data
    risk_threshold = np.std(enhanced_s[120:123]) + 0.5  # Example based on std deviation
    trend_threshold = 0.3  # Example based on historical performance

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for risky BUY signals
        reward += 10 * (1 - risk_level)  # Mild positive for SELL signals
        return np.clip(reward, -100, 100)  # Early exit

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20  # Positive reward for bullish momentum
        else:
            reward += 20  # Positive reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]