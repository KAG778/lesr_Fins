import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Rate of Change (ROC) over the last 14 days
    roc = (s[0] - s[84]) / s[84] if s[84] != 0 else 0  # Compare day 19 to day 5
    features.append(roc)

    # Feature 2: Average Volume over the last 10 days
    avg_volume = np.mean(s[4:120:6][-10:]) if len(s[4:120:6]) >= 10 else 0
    features.append(avg_volume)

    # Feature 3: Z-score of the last 14 closing prices for mean reversion
    closing_prices = s[0:120:6]
    mean_price = np.mean(closing_prices[-14:]) 
    std_price = np.std(closing_prices[-14:]) 
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    features.append(z_score)

    # Feature 4: Average True Range (ATR) to measure volatility
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))) if len(high_prices) > 1 else 0
    features.append(atr)

    # Feature 5: Exponential Moving Average (EMA) of closing prices over the last 14 days
    def calculate_ema(prices, period=14):
        if len(prices) < period:
            return 0
        k = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * k + ema
        return ema

    ema = calculate_ema(closing_prices[-14:])
    features.append(ema)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate relative thresholds for risk
    historical_risk_levels = enhanced_s[123:]  # Extract features for calculation
    mean_risk_level = np.mean(historical_risk_levels)
    std_risk_level = np.std(historical_risk_levels)
    
    high_risk_threshold = mean_risk_level + 1.5 * std_risk_level
    medium_risk_threshold = mean_risk_level + 0.5 * std_risk_level

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > medium_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= medium_risk_threshold:
        reward += 25 * np.sign(trend_direction)  # Reward for alignment with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > np.mean(np.std(enhanced_s[123:])):  # Dynamic threshold based on historical volatility
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure the reward is within the bounds of [-100, 100]