import numpy as np

def revise_state(s):
    closing_prices = s[0::6][:20]  # Extract closing prices
    volumes = s[4::6][:20]          # Extract trading volumes

    # Feature 1: Price momentum (current close vs previous close)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Average Volume Change (current volume vs previous volume)
    avg_volume_change = (volumes[-1] - np.mean(volumes[-5:])) / (np.mean(volumes[-5:]) + 1e-10)

    # Feature 3: Historical Volatility (standard deviation of returns over the last 20 days)
    returns = np.diff(closing_prices) / (closing_prices[:-1] + 1e-10)
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0

    # Feature 4: Relative Strength Index (RSI)
    gains = np.maximum(0, np.diff(closing_prices))
    losses = np.abs(np.minimum(0, np.diff(closing_prices)))
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Feature 5: Recent drawdown (percentage change from peak)
    recent_peak = np.max(closing_prices)
    recent_drawdown = (recent_peak - closing_prices[-1]) / (recent_peak + 1e-10)

    features = [price_momentum, avg_volume_change, historical_volatility, rsi, recent_drawdown]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    price_momentum = features[0]
    avg_volume_change = features[1]
    historical_volatility = features[2]
    rsi = features[3]
    recent_drawdown = features[4]

    # Initialize reward variable
    reward = 0.0

    # Calculate thresholds based on historical data (mean and std dev)
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
    elif abs(trend_direction) <= trend_threshold and risk_level < 0.3:
        if rsi < 30:  # Oversold
            reward += 15  # Positive reward for potential buy
        elif rsi > 70:  # Overbought
            reward += 15  # Positive reward for potential sell
        else:
            reward -= 5  # Penalize for breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds
    return np.clip(reward, -100, 100)