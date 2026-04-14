import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extracting OHLCV data
    closing_prices = s[::6]  # Closing prices
    opening_prices = s[1::6]  # Opening prices
    high_prices = s[2::6]  # High prices
    low_prices = s[3::6]  # Low prices
    volumes = s[4::6]  # Trading volumes

    # Feature 1: Daily Returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0  # Average daily return

    # Feature 2: Volatility (using standard deviation of returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0  # Standard deviation of returns

    # Feature 3: Price Momentum (difference between the latest and the earliest closing price)
    price_momentum = closing_prices[-1] - closing_prices[0]  # Recent price movement

    # Returning only the computed features
    features = [avg_daily_return, volatility, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    avg_daily_return = features[0]
    volatility = features[1]
    price_momentum = features[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if avg_daily_return > 0:  # BUY-aligned feature
            reward = np.random.uniform(-50, -30)
        else:  # SELL-aligned feature
            reward = np.random.uniform(5, 10)
    elif risk_level > 0.4:
        if avg_daily_return > 0:  # BUY-aligned feature
            reward = np.random.uniform(-20, -10)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and avg_daily_return > 0:  # Uptrend & positive return
            reward = np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < -0.3 and avg_daily_return < 0:  # Downtrend & negative return
            reward = np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if avg_daily_return < 0:  # Oversold situation
            reward = np.random.uniform(5, 15)  # Positive reward for buying
        elif avg_daily_return > 0:  # Overbought situation
            reward = np.random.uniform(5, 15)  # Positive reward for selling
        else:
            reward = np.random.uniform(-5, 5)  # Neutral reward

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return float(reward)