import numpy as np

def revise_state(s):
    # Extract relevant price and volume data
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    n_days = 20
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    daily_returns = np.append(0, daily_returns)  # Padding for alignment

    # Feature 1: Current Price Momentum (current closing price - previous closing price)
    momentum = closing_prices[-1] - closing_prices[-2]  # Last day momentum

    # Feature 2: Price Range (high - low) over the last n_days
    price_range = np.max(high_prices[-n_days:]) - np.min(low_prices[-n_days:])

    # Feature 3: 20-day Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns[-n_days:]) if len(daily_returns) >= n_days else 0

    # Feature 4: Average True Range (ATR) to capture market volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-n_days:]) if len(true_ranges) >= n_days else 0

    # Feature 5: Volume Change Percentage (current volume vs. average volume over the last n_days)
    avg_volume = np.mean(volumes[-n_days:]) if len(volumes) >= n_days else 0
    volume_change_percentage = (volumes[-1] - avg_volume) / avg_volume if avg_volume > 0 else 0

    features = [momentum, price_range, volatility, atr, volume_change_percentage]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Determine historical volatility from the features
    historical_std = np.std(enhanced_s[123:])  # Use the features as a proxy for historical std
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)    # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (if the risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        else:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features
        reward -= np.random.uniform(5, 10)   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds