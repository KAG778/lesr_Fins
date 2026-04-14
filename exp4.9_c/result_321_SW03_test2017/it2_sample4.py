import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
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

    # Feature 2: Average True Range (ATR) for volatility measure (14-day)
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 0

    # Feature 3: Z-score of daily returns to adapt to market conditions
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if n > 1 else np.array([0])
    z_score = (np.mean(daily_returns) - np.mean(daily_returns[-14:])) / (np.std(daily_returns[-14:]) if np.std(daily_returns[-14:]) != 0 else 1)

    # Feature 4: Price Momentum (last price - moving average)
    moving_average = np.mean(closing_prices[-20:]) if n >= 20 else closing_prices[-1]
    price_momentum = closing_prices[-1] - moving_average

    features = [rsi, atr, z_score, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    rsi = features[0]
    atr = features[1]
    z_score = features[2]
    price_momentum = features[3]

    # Calculate relative thresholds based on historical data
    historical_std = np.std(features) if len(features) > 0 else 1.0
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std

    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -40 if price_momentum > 0 else 10  # Strong negative for risky BUY, mild positive for SELL
    elif risk_level > medium_risk_threshold:
        reward += -10 if price_momentum > 0 else 0

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        if trend_direction > 0 and price_momentum > 0:  # Uptrend aligned with positive momentum
            reward += 20  # Positive reward for correct trend-following
        elif trend_direction < 0 and price_momentum < 0:  # Downtrend aligned with negative momentum
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if z_score < -1:  # Oversold situation
            reward += 15  # Reward for mean-reversion buying
        elif z_score > 1:  # Overbought situation
            reward += -15  # Penalize chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range