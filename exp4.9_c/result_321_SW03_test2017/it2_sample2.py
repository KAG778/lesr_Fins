import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    n = len(closing_prices)

    # Feature 1: 14-day Relative Strength Index (RSI)
    def compute_rsi(prices, window=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-window:]) if len(gain) >= window else np.mean(gain) if len(gain) > 0 else 0
        avg_loss = np.mean(loss[-window:]) if len(loss) >= window else np.mean(loss) if len(loss) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    # Feature 2: Average True Range (ATR) for volatility
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 0

    # Feature 3: Price Momentum (closing price change)
    price_momentum = closing_prices[-1] - closing_prices[-2] if n > 1 else 0

    # Feature 4: Trend Strength (momentum over last 5 days)
    trend_strength = np.sum(np.diff(closing_prices[-5:])) if n > 5 else 0

    # Feature 5: Crisis Indicator (detecting extreme volatility)
    recent_volatility = np.std(np.diff(closing_prices[-20:]))  # 20-day rolling volatility
    crisis_indicator = 1 if recent_volatility > np.mean(np.std(np.diff(closing_prices[-100:]))) else 0

    # Return all features as a 1D numpy array
    features = [rsi, atr, price_momentum, trend_strength, crisis_indicator]
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
    price_momentum = features[2]
    trend_strength = features[3]
    crisis_indicator = features[4]

    # Initialize reward
    reward = 0.0

    # Calculate thresholds based on historical volatility
    historical_std = np.std(features[:20]) if len(features) > 0 else 1.0
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std

    # Priority 1: Risk Management
    if risk_level > high_risk_threshold:
        reward = np.random.uniform(-50, -30) if price_momentum > 0 else np.random.uniform(5, 10)  # Negative for risky BUY, positive for SELL
    elif risk_level > medium_risk_threshold:
        reward = np.random.uniform(-20, -10) if price_momentum > 0 else 0  # Moderate negative for risky BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        if trend_direction > 0 and price_momentum > 0:  # Uptrend aligned with positive momentum
            reward += np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < 0 and price_momentum < 0:  # Downtrend aligned with negative momentum
            reward += np.random.uniform(10, 20)  # Positive reward

    # Priority 3: Sideways/Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold condition
            reward += np.random.uniform(5, 15)  # Positive reward for mean-reversion buying
        elif rsi > 70:  # Overbought condition
            reward += np.random.uniform(-15, -5)  # Penalize chasing breakouts

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range