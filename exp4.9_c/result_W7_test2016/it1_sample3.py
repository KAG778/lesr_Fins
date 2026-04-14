import numpy as np

def revise_state(s):
    # Extract relevant price and volume data
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    # Calculate daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    daily_returns = np.append(0, daily_returns)  # Padding for alignment

    # Feature 1: Average Daily Return over the last 20 days
    avg_daily_return = np.mean(daily_returns)

    # Feature 2: Volatility (standard deviation of returns over the last 20 days)
    volatility = np.std(daily_returns)

    # Feature 3: Current Price Momentum (current closing price - previous closing price)
    momentum = closing_prices[-1] - closing_prices[-2]  # Last day momentum

    # Feature 4: Price Range (high - low) over the last 20 days
    price_range = np.max(high_prices) - np.min(low_prices)

    # Feature 5: 20-day Volume Moving Average
    volume_ma = np.mean(volumes[-20:])

    # Feature 6: Average True Range (ATR) for volatility measure
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-20:]) if len(true_ranges) >= 20 else np.nan

    features = [avg_daily_return, volatility, momentum, price_range, volume_ma, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Determine thresholds based on historical data
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

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > trend_threshold:
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)   # Reward for mean-reversion features
        reward -= np.random.uniform(5, 10)    # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds