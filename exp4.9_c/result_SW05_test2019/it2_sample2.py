import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    features = []

    # 1. Exponential Moving Average (EMA) for recent prices
    def calculate_ema(prices, period=10):
        if len(prices) < period:
            return 0
        alpha = 2 / (period + 1)
        ema = prices[0]  # Start with the first price
        for price in prices[1:period]:
            ema = (price - ema) * alpha + ema
        return ema

    recent_ema = calculate_ema(closing_prices[-10:])
    features.append(recent_ema)

    # 2. Rate of Change (ROC) of closing prices
    if len(closing_prices) > 5:
        roc = (closing_prices[0] - closing_prices[5]) / closing_prices[5]
    else:
        roc = 0
    features.append(roc)

    # 3. Standard Deviation of Volume (for volatility context)
    if len(volumes) > 5:
        vol_std = np.std(volumes[-5:])
    else:
        vol_std = 0
    features.append(vol_std)

    # 4. Z-score of Closing Prices to measure current price relative to historical mean
    historical_mean = np.mean(closing_prices[-20:]) if len(closing_prices) > 20 else 0
    historical_std = np.std(closing_prices[-20:]) if len(closing_prices) > 20 else 0
    z_score_price = (closing_prices[0] - historical_mean) / historical_std if historical_std > 0 else 0
    features.append(z_score_price)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds based on the standard deviation of features
    historical_std = np.std(enhanced_s[123:])  # Assuming features start from index 123
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(40, 60)  # Strong negative reward for BUY
    elif risk_level > risk_threshold_moderate:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        features = enhanced_s[123:]
        if trend_direction > 0:  # Uptrend
            if features[1] > 0:  # Aligning with positive momentum
                reward += 20  # Strong positive reward
        elif trend_direction < 0:  # Downtrend
            if features[1] < 0:  # Aligning with negative momentum
                reward += 20  # Strong positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= trend_threshold and risk_level < 0.3:
        features = enhanced_s[123:]
        if features[3] < -1:  # Oversold condition (applying z-score)
            reward += 10  # Reward for buying in oversold
        elif features[3] > 1:  # Overbought condition
            reward += 10  # Reward for selling in overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds