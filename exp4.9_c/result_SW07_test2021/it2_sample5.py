import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    high_prices = s[2::6]      # High prices
    low_prices = s[3::6]       # Low prices

    # Feature 1: Exponential Moving Average (EMA) of closing prices (20 days)
    ema = np.mean(closing_prices[-20:])  # Using a simple mean for now, can be replaced with EMA calculation

    # Feature 2: Average Volume over the last 20 days
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1  # Avoid division by zero

    # Feature 3: Price Change Percentage from the last closing price
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 4: Z-score of closing prices based on historical mean and std dev
    historical_mean = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    historical_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1  # Avoid division by zero
    z_score = (closing_prices[-1] - historical_mean) / historical_std if historical_std != 0 else 0

    features = [ema, average_volume, price_change_pct, z_score]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Calculate relative thresholds based on historical data (using std of features)
    historical_std = np.std(enhanced_s[123:])  # Features start at index 123
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY signals
        reward += np.random.uniform(5, 10)    # Mild positive for SELL signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds