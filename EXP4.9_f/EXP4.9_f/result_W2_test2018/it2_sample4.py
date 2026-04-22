import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices for the last 20 days
    closing_prices = s[0::6]  # OHLCV state, closing prices every 6th element
    daily_returns = np.zeros(len(closing_prices))
    
    # Calculate daily returns
    for i in range(1, len(closing_prices)):
        if closing_prices[i - 1] > 0:  # Avoid division by zero
            daily_returns[i] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]

    # Feature 1: Average daily return
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)

    # Feature 2: Historical Volatility (standard deviation of daily returns)
    historical_volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
    features.append(historical_volatility)

    # Feature 3: Price momentum (current close vs. close 5 days ago)
    price_momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if len(closing_prices) > 5 and closing_prices[5] > 0 else 0
    features.append(price_momentum)

    # Feature 4: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 5: Extreme Price Movement (to identify crisis)
    extreme_movement = np.max(np.abs(daily_returns)) if len(daily_returns) > 1 else 0
    features.append(extreme_movement)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    avg_daily_return = features[0]
    historical_volatility = features[1]
    price_momentum = features[2]
    rsi = features[3]
    extreme_movement = features[4]

    reward = 0.0

    # Calculate dynamic risk thresholds
    risk_threshold_high = 0.7 * historical_volatility if historical_volatility != 0 else 0
    risk_threshold_moderate = 0.4 * historical_volatility if historical_volatility != 0 else 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if avg_daily_return > 0:  # BUY-aligned feature
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:  # SELL-aligned feature
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > risk_threshold_moderate:
        if avg_daily_return > 0:  # BUY-aligned feature
            reward = np.random.uniform(-20, -10)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0 and price_momentum > 0:  # Uptrend and upward momentum
            reward += 10  # Positive reward
        elif trend_direction < 0 and price_momentum < 0:  # Downtrend and downward momentum
            reward += 10  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold condition
            reward += 10  # Positive reward for mean-reversion buy
        elif rsi > 70:  # Overbought condition
            reward -= 10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds