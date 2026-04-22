import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes
    
    features = []

    # Feature 1: Relative Price Change (current price vs. 5 days ago)
    if len(closing_prices) > 5:
        rel_price_change = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if closing_prices[5] != 0 else 0
        features.append(rel_price_change)
    else:
        features.append(0)

    # Feature 2: Volume Change (current volume vs. 5 days ago)
    if len(trading_volumes) > 5:
        rel_volume_change = (trading_volumes[0] - trading_volumes[5]) / trading_volumes[5] if trading_volumes[5] != 0 else 0
        features.append(rel_volume_change)
    else:
        features.append(0)

    # Feature 3: 5-day Moving Average Convergence Divergence (MACD)
    short_term_ma = np.mean(closing_prices[:5]) if len(closing_prices) >= 5 else 0
    long_term_ma = np.mean(closing_prices[5:10]) if len(closing_prices) >= 10 else 0
    macd = short_term_ma - long_term_ma
    features.append(macd)

    # Feature 4: Z-Score of recent returns for risk assessment
    recent_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    if len(recent_returns) > 5:
        z_score = (np.mean(recent_returns) - np.mean(recent_returns[-5:])) / np.std(recent_returns[-5:]) if np.std(recent_returns[-5:]) != 0 else 0
        features.append(z_score)
    else:
        features.append(0)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Bullish trend based on relative price change
            reward -= np.random.uniform(30, 50)  # Strong negative for buying in high-risk
        else:  # Bearish or neutral trend
            reward += np.random.uniform(5, 10)  # Small positive for selling
    elif risk_level > 0.4:
        if features[0] > 0:
            reward -= np.random.uniform(10, 20)  # Moderate negative for bullish in moderate risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10 * features[0]  # Positive reward for long positions in uptrend
        else:
            reward -= 10 * features[0]  # Negative reward for long positions in downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition
            reward += 15.0  # Positive for buying in oversold
        elif features[0] > 0.1:  # Overbought condition
            reward -= 15.0  # Negative for buying in overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range