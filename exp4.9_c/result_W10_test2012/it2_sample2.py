import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices, high prices, and low prices from the state
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    
    # Feature 1: Average True Range (ATR) for volatility
    def compute_atr(prices, period=14):
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - prices[:-1]), 
                                   np.abs(low_prices[1:] - prices[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr) if len(tr) > 0 else 0
        return atr

    atr = compute_atr(closing_prices)

    # Feature 2: Rate of Change (ROC) for momentum
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) >= 15 and closing_prices[-15] != 0 else 0

    # Feature 3: Mean daily return over the last 20 days
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(mean_daily_return)

    # Feature 4: Historical volatility as standard deviation of daily returns over the last 20 days
    historical_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(historical_volatility)

    # Feature 5: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rs = avg_gain / avg_loss if avg_loss > 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_risk = np.mean(enhanced_s[123])  # Mean of the features as a proxy for risk
    historical_std = np.std(enhanced_s[123])    # Standard deviation of features

    high_risk_threshold = historical_risk + 1.5 * historical_std
    low_risk_threshold = historical_risk - 1.5 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
        return max(-100, reward)  # Immediate return to prioritize risk management
    elif risk_level > low_risk_threshold:
        reward += 20 if enhanced_s[123][1] < 0 else -50  # Mild positive for SELL, strong negative for BUY

    # Extract features
    features = enhanced_s[123:]

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0 and features[2] > 0:  # Upward momentum
            reward += 40  # Strong positive reward for upward alignment
        elif trend_direction < 0 and features[2] < 0:  # Downward momentum
            reward += 40  # Strong positive reward for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        if features[3] < 30:  # RSI indicates oversold
            reward += 30  # Buy signal when oversold
        elif features[3] > 70:  # RSI indicates overbought
            reward -= 30  # Strong negative for buying in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward