import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0:120:6]  # Every 6th element starting from index 0 is the closing price
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Price Change Percentage from 5 days ago
    if closing_prices[15] != 0:  # Avoid division by zero
        price_change_percentage = (closing_prices[19] - closing_prices[15]) / closing_prices[15]
    else:
        price_change_percentage = 0
    features.append(price_change_percentage)

    # Feature 2: Historical Volatility (20-day)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns) * np.sqrt(20)  # Annualized volatility
    features.append(historical_volatility)

    # Feature 3: Relative Strength Index (RSI)
    rsi_period = 14
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    avg_gain = np.mean(gains[-rsi_period:]) if len(gains) >= rsi_period else 0
    avg_loss = np.mean(losses[-rsi_period:]) if len(losses) >= rsi_period else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 4: Momentum (Rate of Change)
    momentum = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if closing_prices[-5] != 0 else 0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 15)    # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        features = enhanced_s[123:]  # Retrieve computed features
        if trend_direction > 0:  # Uptrend
            reward += 10 * features[3]  # Feature 3: Momentum
        elif trend_direction < 0:  # Downtrend
            reward += 10 * (1 - features[3])  # Inverse momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        features = enhanced_s[123:]
        if features[2] < 30:  # Assuming RSI < 30 is oversold
            reward += 15.0  # Reward for mean-reversion BUY
        elif features[2] > 70:  # Assuming RSI > 70 is overbought
            reward += 15.0  # Reward for mean-reversion SELL

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > np.mean(enhanced_s[123:]) * 1.5:  # Relative measure
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range