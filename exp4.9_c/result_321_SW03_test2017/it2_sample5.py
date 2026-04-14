import numpy as np

def revise_state(s):
    # Extracting closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0 (closing prices)
    
    n = len(closing_prices)

    # Feature 1: 14-day Relative Strength Index (RSI)
    def compute_rsi(prices, window=14):
        if len(prices) < window:
            return np.zeros_like(prices)[-1]  # Return 0 if not enough data
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-window:])  # Use only the last 'window' days
        avg_loss = np.mean(loss[-window:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    # Feature 2: Average True Range (ATR) to measure volatility
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # ATR over the last 14 days

    # Feature 3: Z-Score of daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if n > 1 else np.array([0])
    z_score = (np.mean(daily_returns) - np.mean(daily_returns[-14:])) / (np.std(daily_returns[-14:]) if np.std(daily_returns[-14:]) != 0 else 1)

    # Feature 4: Crisis Indicator based on recent volatility
    recent_volatility = np.std(daily_returns[-20:])  # Volatility of the last 20 days
    crisis_indicator = 1 if recent_volatility > np.mean(np.std(daily_returns[-100:])) else 0  # High volatility detection

    # Feature 5: Price Momentum (current price - moving average)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]  # 20-day MA
    price_momentum = closing_prices[-1] - moving_average

    features = [rsi, atr, z_score, crisis_indicator, price_momentum]
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
    crisis_indicator = features[3]
    price_momentum = features[4]

    # Initialize reward
    reward = 0.0

    # Calculate relative thresholds using historical std
    historical_std = np.std(features) if len(features) > 0 else 1.0
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std

    # Priority 1: Risk Management
    if risk_level > high_risk_threshold:
        if price_momentum > 0:  # BUY-aligned feature
            reward = np.random.uniform(-50, -30)  # Strong negative for risky BUY
        else:  # SELL-aligned feature
            reward = np.random.uniform(5, 10)  # Mild positive for SELL
    elif risk_level > medium_risk_threshold:
        if price_momentum > 0:  # BUY signal
            reward = np.random.uniform(-20, -10)  # Moderate negative for risky BUY

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        if trend_direction > 0 and price_momentum > 0:  # Uptrend aligned with positive momentum
            reward += np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < 0 and price_momentum < 0:  # Downtrend aligned with negative momentum
            reward += np.random.uniform(10, 20)  # Positive reward

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if z_score < -1:  # Oversold condition
            reward += np.random.uniform(5, 15)  # Positive for mean-reversion buying
        elif z_score > 1:  # Overbought condition
            reward += np.random.uniform(-15, -5)  # Negative for chasing breakouts

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range