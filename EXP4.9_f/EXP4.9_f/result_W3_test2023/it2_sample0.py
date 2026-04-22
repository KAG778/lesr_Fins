import numpy as np

def revise_state(s):
    # Extract relevant data from the state
    closing_prices = s[0::6][:20]  # Extract closing prices
    volumes = s[4::6][:20]          # Extract trading volumes

    # Feature 1: Price Momentum (current close vs previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) >= 2 else 0

    # Feature 2: Average Volume Change (current volume vs average volume of the last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    volume_change = (volumes[-1] - avg_volume) / (avg_volume + 1e-10)  # Normalized volume change

    # Feature 3: Historical Volatility (standard deviation of returns over the last 20 days)
    returns = np.diff(closing_prices) / (closing_prices[:-1] + 1e-10)  # Daily returns
    historical_volatility = np.std(returns) if len(returns) > 0 else 0

    # Feature 4: Recent Drawdown (percentage from peak)
    recent_peak = np.max(closing_prices)
    recent_drawdown = (recent_peak - closing_prices[-1]) / (recent_peak + 1e-10)

    # Feature 5: 14-period Relative Strength Index (RSI)
    gains = np.maximum(0, np.diff(closing_prices))
    losses = np.abs(np.minimum(0, np.diff(closing_prices)))
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    features = [price_momentum, volume_change, historical_volatility, recent_drawdown, rsi]
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
    rsi = features[4]

    # Initialize reward variable
    reward = 0.0

    # Calculate relative thresholds based on historical data (e.g., mean and std dev)
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if price_momentum > 0:  # Strong bullish signal
            reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        else:
            reward += np.random.uniform(5, 10)  # Mild positive for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_high:
        if trend_direction > 0 and price_momentum > 0:  # Bullish trend and bullish signal
            reward += 20  # Positive reward for correct bullish bet
        elif trend_direction < 0 and price_momentum < 0:  # Bearish trend and bearish signal
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= trend_threshold and risk_level < risk_threshold_medium:
        if rsi < 30:  # Oversold
            reward += 15  # Positive reward for potential buy
        elif rsi > 70:  # Overbought
            reward += 15  # Positive reward for potential sell
        else:
            reward -= 5  # Penalize for breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds
    return np.clip(reward, -100, 100)