import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Rate of Change (ROC) over the last 14 days
    roc = (s[0] - s[84]) / s[84] if s[84] != 0 else 0  # Closing price change from day 6 to day 20
    
    # Feature 2: Exponential Moving Average (EMA) of Closing Prices over the last 14 days
    def calculate_ema(prices, span=14):
        weights = np.exp(np.linspace(-1, 0, span))
        weights /= weights.sum()
        return np.dot(prices[-span:], weights)

    ema = calculate_ema(s[0:120:6])  # Last 20 closing prices for EMA

    # Feature 3: Z-score of the last 14 closing prices for mean reversion
    closing_prices = s[0:120:6][-14:]  # Last 14 closing prices
    mean_price = np.mean(closing_prices)
    std_price = np.std(closing_prices)
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    features.append(z_score)

    # Feature 4: Average True Range (ATR) over the last 14 days
    high_prices = s[2:120:6]  # High prices
    low_prices = s[3:120:6]   # Low prices
    atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1]))))
    features.append(atr)

    return np.array([roc, ema, z_score, atr])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(enhanced_s[123:])  # Based on features
    high_risk_threshold = 0.7 * historical_volatility
    medium_risk_threshold = 0.4 * historical_volatility

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY signals in high-risk environments
        reward += 10   # Mild positive reward for SELL signals
    elif risk_level > medium_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= medium_risk_threshold:
        if trend_direction > 0:
            reward += 20  # Positive reward for upward momentum
        else:
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)