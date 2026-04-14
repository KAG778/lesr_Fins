import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    
    # Feature 1: Mean and Standard Deviation of daily returns over the last 20 days
    if len(daily_returns) >= 20:
        mean_return = np.mean(daily_returns[-20:])
        volatility = np.std(daily_returns[-20:])
    else:
        mean_return = 0
        volatility = 0
    features.extend([mean_return, volatility])

    # Feature 2: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss if avg_loss > 0 else 1)))  # Avoid division by zero
    features.append(rsi)

    # Feature 3: Average True Range (ATR) for volatility
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]   # Extract low prices
    true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:])
    true_ranges = np.maximum(true_ranges, low_prices[1:] - closing_prices[:-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    features.append(atr)

    # Feature 4: Rate of Change (ROC) for momentum
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) >= 15 and closing_prices[-15] != 0 else 0
    features.append(roc)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Calculate historical thresholds for risk management
    historical_risk_level = np.mean(enhanced_s[123])  # Average of feature values as proxy for risk
    historical_std_risk = np.std(enhanced_s[123])  # Historical standard deviation for risk

    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_risk_level + 1.5 * historical_std_risk:  # High risk
        reward -= 50  # Strong negative for BUY-aligned signals
        return max(-100, reward)  # Return immediately
    elif risk_level > historical_risk_level + 0.5 * historical_std_risk:  # Moderate risk
        reward -= 20  # Moderate negative for BUY signals

    # Extract features for further decision-making
    features = enhanced_s[123:]

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < historical_risk_level:
        if trend_direction > 0:  # Upward momentum
            reward += 30  # Strong reward for alignment with upward trend
        else:  # Downward momentum
            reward += 30  # Strong reward for alignment with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < historical_risk_level:
        if features[2] < 30:  # RSI indicates oversold
            reward += 30  # Reward for buying in an oversold market
        elif features[2] > 70:  # RSI indicates overbought
            reward -= 30  # Penalize for buying in an overbought market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < historical_risk_level:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward