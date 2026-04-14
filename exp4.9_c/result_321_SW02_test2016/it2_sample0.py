import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Every 6th element is a closing price
    volumes = s[4:120:6]          # Extract trading volumes

    # Feature 1: 10-day Momentum
    momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0
    features.append(momentum)

    # Feature 2: Exponential Moving Average (EMA) of closing prices (last 10 days)
    weights = np.exp(np.linspace(-1., 0., 10))
    weights /= weights.sum()
    ema = np.dot(weights, closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    features.append(ema)

    # Feature 3: Relative Strength Index (RSI) (14-day period)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50.0  # Neutral RSI when not enough data
    features.append(rsi)

    # Feature 4: Average True Range (ATR)
    high_prices = s[1:120:6]  # High prices for 20 days
    low_prices = s[2:120:6]   # Low prices for 20 days
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    mean_risk = np.mean([0.5, 0.7])  # Example mean risk level based on historical performance
    std_risk = 0.2  # Example std for risk level
    risk_threshold = mean_risk + 1 * std_risk  # Example threshold based on std deviation

    mean_trend = 0.3  # Example mean trend level
    std_trend = 0.1  # Example std for trend level
    trend_threshold = mean_trend + 1 * std_trend  # Example threshold based on std deviation

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative penalty for buying in high risk
        reward += 10 * (1 - risk_level)  # Mild positive for selling
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > 0.4:
        reward -= 20  # Moderate penalty for buying in elevated risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20  # Positive reward for bullish alignment
        elif trend_direction < 0:
            reward += 20  # Positive reward for bearish alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]