import numpy as np

def revise_state(s):
    features = []
    
    # Closing prices
    closes = s[0::6]  # Closing prices
    daily_returns = np.diff(closes) / closes[:-1]  # Daily returns
    
    # Feature 1: Mean Daily Return
    mean_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
    features.append(mean_return)
    
    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
    features.append(volatility)
    
    # Feature 3: Relative Strength Index (RSI) for the last 14 days
    window_length = 14
    if len(daily_returns) > window_length:
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)

        avg_gain = np.mean(gains[-window_length:])
        avg_loss = np.mean(losses[-window_length:])
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)
    else:
        features.append(50)  # Neutral RSI

    # Feature 4: Sharpe Ratio (using last 20 days)
    if len(daily_returns) >= 20:
        sharpe_ratio = np.mean(daily_returns[-20]) / (np.std(daily_returns[-20]) if np.std(daily_returns[-20]) != 0 else 1)
        features.append(sharpe_ratio)
    else:
        features.append(0)  # Default value for Sharpe Ratio

    # Feature 5: Drawdown ratio (last 20 days)
    max_drawdown = np.max(np.maximum.accumulate(closes[-20]) - closes[-20]) / (np.max(closes[-20]) if np.max(closes[-20]) != 0 else 1)
    features.append(max_drawdown)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    historical_std = np.std(enhanced_s[123:])  # Use historical std as a dynamic threshold
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY signals
        reward += np.random.uniform(5, 10)  # Mild positive for SELL signals
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for bullish features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward for mean-reversion strategy
        reward += 10  # Example positive reward for mean-reversion alignment

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the specified range
    return float(np.clip(reward, -100, 100))