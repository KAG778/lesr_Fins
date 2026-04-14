import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    features = []

    # Feature 1: Exponential Moving Average (EMA) of closing prices
    def calculate_ema(prices, period=14):
        if len(prices) < period:
            return 0
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.dot(weights, prices[-period:])

    ema_value = calculate_ema(closing_prices)
    features.append(ema_value)

    # Feature 2: Percentage Price Oscillator (PPO)
    if len(closing_prices) > 26:  # Minimum length for 12 and 26 EMA
        ema12 = calculate_ema(closing_prices, period=12)
        ema26 = calculate_ema(closing_prices, period=26)
        ppo = (ema12 - ema26) / ema26 * 100 if ema26 != 0 else 0
    else:
        ppo = 0
    features.append(ppo)

    # Feature 3: Average Volume over the last 10 days
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0.0
    recent_volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0.0
    features.append(recent_volume_change)

    # Feature 4: Standard Deviation of Closing Prices (Volatility)
    price_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    features.append(price_volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:]) if len(enhanced_s[123:]) > 0 else 1  # Avoid division by zero
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned
        reward += np.random.uniform(5, 10)    # Mild positive for SELL-aligned

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:  # Uptrend
            reward += 20  # Positive reward for upward alignment
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 20  # Positive reward for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Assuming oversold situation (mean reversion BUY)
            reward += 15  # Reward for mean-reversion BUY
        elif enhanced_s[123] > 0:  # Overbought situation (mean reversion SELL)
            reward -= 15  # Penalize for mean-reversion SELL

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]