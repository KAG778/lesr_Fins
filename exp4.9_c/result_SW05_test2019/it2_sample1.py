import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    features = []

    # 1. Price Momentum (current close - average close of the last 5 days)
    if len(closing_prices) > 5:
        momentum = closing_prices[0] - np.mean(closing_prices[1:6])
    else:
        momentum = 0
    features.append(momentum)

    # 2. Z-Score of Closing Prices (to assess how current price compares to historical average)
    historical_mean = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    historical_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1  # Avoid division by zero
    z_score_price = (closing_prices[0] - historical_mean) / historical_std if historical_std > 0 else 0
    features.append(z_score_price)

    # 3. Average True Range (ATR) as a volatility measure
    def calculate_atr(prices):
        true_ranges = []
        for i in range(1, len(prices)):
            high = s[2 + i * 6]
            low = s[3 + i * 6]
            close_prev = prices[i - 1]
            true_ranges.append(max(high - low, abs(high - close_prev), abs(low - close_prev)))
        return np.mean(true_ranges[-14:]) if len(true_ranges) > 13 else 0
    
    atr = calculate_atr(closing_prices)
    features.append(atr)

    # 4. Rate of Change (ROC) to capture momentum shifts over a longer horizon
    if len(closing_prices) > 5:
        roc = ((closing_prices[0] - closing_prices[5]) / closing_prices[5]) * 100  # ROC in percentage
    else:
        roc = 0
    features.append(roc)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk
    historical_std = np.std(enhanced_s[0:120])  # Using the raw state for std calculation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    momentum_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(40, 60)  # Strong negative reward for BUY
        reward += np.random.uniform(5, 15)    # Mild positive reward for SELL
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > momentum_threshold and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Strong positive reward for aligning with upward momentum
        elif trend_direction < 0:  # Downtrend
            reward += 20  # Strong positive reward for aligning with downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= momentum_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds