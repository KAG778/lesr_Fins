import numpy as np

def revise_state(s):
    # Extracting closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0 (closing prices)
    n = len(closing_prices)

    # Feature 1: 14-day Relative Strength Index (RSI)
    def compute_rsi(prices, window=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain) if len(gain) > 0 else 0
        avg_loss = np.mean(loss) if len(loss) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    # Feature 2: Average True Range (ATR) for volatility (20-day)
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-20:]) if len(tr) >= 20 else 0

    # Feature 3: Z-score of daily returns to adapt to market conditions
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if n > 1 else np.array([0])
    z_score = (np.mean(daily_returns) - np.mean(daily_returns[-14:])) / (np.std(daily_returns[-14:]) if np.std(daily_returns[-14:]) != 0 else 1)

    # Feature 4: Price Momentum (the difference between the latest closing price and the moving average)
    moving_average = np.mean(closing_prices[-20:]) if n >= 20 else closing_prices[-1]  # 20-day moving average
    price_momentum = closing_prices[-1] - moving_average if n > 0 else 0

    # Return computed features
    features = [rsi, atr, z_score, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    # Extract features
    rsi = features[0]
    atr = features[1]
    z_score = features[2]
    price_momentum = features[3]

    # Initialize reward
    reward = 0.0

    # Dynamic risk thresholds based on historical volatility
    historical_std = np.std(features) if len(features) > 0 else 1.0
    high_risk_threshold = 0.7 * historical_std
    moderate_risk_threshold = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        if price_momentum > 0:  # BUY-aligned feature
            reward = np.random.uniform(-100, -50)  # Strong negative for risky BUY
        else:  # SELL-aligned feature
            reward = np.random.uniform(10, 20)  # Mild positive for SELL
    elif risk_level > moderate_risk_threshold:
        if price_momentum > 0:  # BUY signal
            reward = np.random.uniform(-30, -10)  # Moderate negative for risky BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < moderate_risk_threshold:
        if trend_direction > 0 and price_momentum > 0:  # Uptrend & positive momentum
            reward += np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < 0 and price_momentum < 0:  # Downtrend & negative momentum
            reward += np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if z_score < -1:  # Oversold condition
            reward += np.random.uniform(5, 15)  # Positive reward for mean-reversion buying
        elif z_score > 1:  # Overbought condition
            reward += np.random.uniform(-15, -5)  # Negative reward for buying

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < moderate_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range