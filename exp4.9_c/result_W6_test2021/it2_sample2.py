import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Rate of Change (ROC) over the last 14 days for price momentum
    roc = (s[0] - s[84]) / s[84] if s[84] != 0 else 0  # Closing price change over the last 14 days
    features.append(roc)

    # Feature 2: Average Volume over the last 10 days
    avg_volume = np.mean(s[4:120:6][-10:]) if len(s[4:120:6]) >= 10 else np.mean(s[4:120:6]) if len(s[4:120:6]) > 0 else 0
    features.append(avg_volume)

    # Feature 3: Z-score of the last 14 closing prices for mean reversion indication
    closing_prices = s[0:120:6]
    recent_prices = closing_prices[-14:]  # Last 14 closing prices
    mean_price = np.mean(recent_prices)
    std_price = np.std(recent_prices)
    z_score = (recent_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    features.append(z_score)

    # Feature 4: Exponential Moving Average (EMA) of closing prices for trend detection
    def calculate_ema(prices, span=14):
        if len(prices) < span:
            return 0
        weights = np.exp(np.linspace(-1, 0, span))
        weights /= weights.sum()
        return np.dot(prices[-span:], weights)

    ema = calculate_ema(closing_prices)
    features.append(ema)

    # Feature 5: Average True Range (ATR) for volatility measurement
    def calculate_atr(high_prices, low_prices, closes, period=14):
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closes[:-1]), 
                                   np.abs(low_prices[1:] - closes[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else 0
        return atr

    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    atr = calculate_atr(high_prices, low_prices, closing_prices)
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(enhanced_s[123:])  # Based on features
    high_risk_threshold = 0.7 * historical_volatility
    medium_risk_threshold = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > medium_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= medium_risk_threshold:
        reward += 20 * np.sign(trend_direction)  # Reward for directional alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)