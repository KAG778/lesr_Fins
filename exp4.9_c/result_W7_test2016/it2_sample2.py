import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: Price Momentum based on the last 10 days
    momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0

    # Feature 2: Average True Range (ATR) to capture market volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-10:]) if len(true_ranges) >= 10 else 0  # Use last 10 days for ATR

    # Feature 3: Moving Average of Volume to capture volume trends
    volume_ma = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0

    # Feature 4: Rate of Change of Closing Prices (momentum indicator)
    roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) > 11 else 0

    # Feature 5: Historical Volatility (Standard Deviation of daily returns over the last 20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0

    features = [momentum, atr, volume_ma, roc, historical_volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical volatility and risk thresholds based on past data
    historical_returns = enhanced_s[0:120][0::6]  # Extract closing prices
    if len(historical_returns) > 1:
        daily_returns = np.diff(historical_returns) / historical_returns[:-1]
        historical_volatility = np.std(daily_returns)
    else:
        historical_volatility = 0

    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_medium = 0.4 * historical_volatility
    trend_threshold = 0.3 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)    # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > 0:
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        else:
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)    # Reward for mean-reversion features
        reward -= np.random.uniform(5, 10)     # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility * 1.5 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds