import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices
    
    # Feature 1: Average True Range (ATR) over the last 14 days
    def compute_atr(prices, period=14):
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)  # Ensure enough data
        return atr

    atr = compute_atr(closing_prices)

    # Feature 2: 14-day moving average of returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    moving_avg_returns = np.mean(daily_returns[-14:]) if len(daily_returns) >= 14 else 0

    # Feature 3: Relative Strength Index (RSI)
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = -np.where(daily_returns < 0, daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    features = [atr, moving_avg_returns, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Get historical volatility and average returns for relative thresholds
    historical_volatility = np.std(enhanced_s[123])  # Assuming features start at 123
    historical_return = np.mean(enhanced_s[123]) 
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -100  # Strong negative reward for BUY-aligned features
        return reward  # Immediate return to prioritize risk management
    elif risk_level > 0.4:
        reward += 20 if enhanced_s[123][1] < 0 else -50  # Mild positive for SELL, strong negative for BUY

    # Extract features
    features = enhanced_s[123:]

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[1] > 0:  # Upward momentum
            reward += 50  # Strong positive reward for bullish momentum
        elif trend_direction < -0.3 and features[1] < 0:  # Downward momentum
            reward += 50  # Strong positive reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # RSI indicates oversold
            reward += 30  # Buy signal when oversold
        elif features[2] > 70:  # RSI indicates overbought
            reward += -30  # Strong negative for buying in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward