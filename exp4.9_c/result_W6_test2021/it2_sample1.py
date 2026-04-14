import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Price Rate of Change (ROC) over the last 14 days
    roc = (s[0] - s[84]) / s[84] if s[84] != 0 else 0  # Current price vs price 14 days ago
    features.append(roc)

    # Feature 2: Average True Range (ATR) for volatility measurement
    def calculate_atr(high_prices, low_prices, close_prices, period=14):
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - close_prices[:-1]), 
                                   np.abs(low_prices[1:] - close_prices[:-1])))
        return np.mean(tr[-period:])

    high_prices = s[2:120:6]   # Extract high prices
    low_prices = s[3:120:6]    # Extract low prices
    atr = calculate_atr(high_prices, low_prices, s[0:120:6])
    features.append(atr)

    # Feature 3: Z-score of the last 14 closing prices for mean reversion
    closing_prices = s[0:120:6][-14:]  # Last 14 closing prices
    mean_price = np.mean(closing_prices)
    std_price = np.std(closing_prices)
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    features.append(z_score)

    # Feature 4: Correlation between price changes and volumes over the last 10 days
    price_changes = np.diff(s[0:120:6])
    volume_changes = np.diff(s[4:120:6])
    correlation = np.corrcoef(price_changes[-10:], volume_changes[-10:])[0, 1] if len(price_changes) > 1 and len(volume_changes) > 1 else 0
    features.append(correlation)

    # Feature 5: Exponential Moving Average (EMA) of closing prices over the last 14 days
    def calculate_ema(prices, span=14):
        weights = np.exp(np.linspace(-1, 0, span))
        weights /= weights.sum()
        return np.dot(prices[-span:], weights)

    ema = calculate_ema(s[0:120:6])
    features.append(ema)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical data
    historical_volatility = np.std(enhanced_s[123:])
    high_risk_threshold = np.mean(enhanced_s[123:]) + 1.5 * historical_volatility
    mid_risk_threshold = np.mean(enhanced_s[123:]) + 0.5 * historical_volatility

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY in high-risk
        reward += 10   # Mild positive reward for SELL
    elif risk_level > mid_risk_threshold:
        reward -= 20  # Moderate negative for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level <= mid_risk_threshold:
        reward += 20 if trend_direction > 0 else 20  # Align with momentum direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > np.mean(enhanced_s[123:]) + 1.5 * np.std(enhanced_s[123:]):
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds [-100, 100]
    return np.clip(reward, -100, 100)