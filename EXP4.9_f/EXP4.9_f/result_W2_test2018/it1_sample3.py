import numpy as np

def revise_state(s):
    # s: 120d raw state, containing 20 days of OHLCV data
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    days = len(closing_prices)

    # Feature 1: Price Movement (current price vs. average of last 20 days)
    average_price = np.mean(closing_prices) if days > 1 else 0
    price_movement = (closing_prices[-1] - average_price) / (average_price if average_price != 0 else 1)

    # Feature 2: Volume Spike (current volume vs. average of last 20 days)
    average_volume = np.mean(volumes) if days > 1 else 1
    volume_spike = (volumes[-1] - average_volume) / average_volume

    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / (avg_loss if avg_loss != 0 else 1)
    rsi = 100 - (100 / (1 + rs))

    features = [price_movement, volume_spike, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Relative thresholds based on historical volatility
    historical_volatility = np.std(enhanced_s[123:])  # Use features for volatility baseline
    high_risk_threshold = 0.7 * historical_volatility
    moderate_risk_threshold = 0.4 * historical_volatility

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += np.random.uniform(-50, -30)  # Strong negative for risky BUY
    elif risk_level > moderate_risk_threshold:
        reward += -20  # Moderate negative for risky BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < moderate_risk_threshold:
        momentum = enhanced_s[123]  # Use the first feature for momentum
        if trend_direction > 0 and momentum > 0:  # Uptrend with positive momentum
            reward += np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < 0 and momentum < 0:  # Downtrend with negative momentum
            reward += np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Oversold condition
            reward += np.random.uniform(5, 15)  # Reward for buying in a mean-reversion
        else:
            reward += np.random.uniform(-10, 0)  # Penalize for chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_volatility and risk_level < moderate_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds