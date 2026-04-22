import numpy as np

def revise_state(s):
    closing_prices = s[0::6][:20]  # Extract closing prices
    volumes = s[4::6][:20]          # Extract trading volumes

    # Feature 1: Price momentum (current close vs previous close)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Average Volume Change (percentage change from previous average)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1e-10
    volume_change = (volumes[-1] - avg_volume) / (avg_volume + 1e-10)  # Relative volume change

    # Feature 3: Historical Volatility (standard deviation of returns over the last 20 days)
    returns = np.diff(closing_prices) / (closing_prices[:-1] + 1e-10)  # Daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0

    # Feature 4: Recent Drawdown (percentage change from recent peak)
    recent_peak = np.max(closing_prices)
    recent_drawdown = (recent_peak - closing_prices[-1]) / (recent_peak + 1e-10)

    # Feature 5: Smoothed Price Momentum (exponential moving average of momentum)
    smoothed_momentum = np.mean([closing_prices[-i] - closing_prices[-i-1] for i in range(1, 6)])  # EMA over last 5 days

    features = [price_momentum, volume_change, historical_volatility, recent_drawdown, smoothed_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    price_momentum = features[0]
    volume_change = features[1]
    historical_volatility = features[2]
    recent_drawdown = features[3]
    smoothed_momentum = features[4]

    # Initialize reward variable
    reward = 0.0

    # Calculate thresholds based on historical data
    historical_std = np.std(features) if np.std(features) > 0 else 1e-6
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        if price_momentum > 0:  # Strong bullish signal
            reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        else:
            reward += np.random.uniform(5, 10)  # Mild positive for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > 0 and price_momentum > 0:  # Bullish trend and bullish signal
            reward += 20  # Positive reward for correct bullish bet
        elif trend_direction < 0 and price_momentum < 0:  # Bearish trend and bearish signal
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if recent_drawdown > 0.1:  # Oversold condition (recent drawdown > 10%)
            reward += 15  # Positive reward for potential buy
        elif recent_drawdown < -0.1:  # Overbought condition (recent drawdown < -10%)
            reward += 15  # Positive reward for potential sell
        else:
            reward -= 5  # Penalize for breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds
    return np.clip(reward, -100, 100)