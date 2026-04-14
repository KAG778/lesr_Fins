import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []
    
    # Reshape the raw state into a 20-day view with 6 features each
    days = s.reshape((20, 6))
    
    # Feature 1: Daily Return Volatility (Standard Deviation of Daily Returns)
    closing_prices = days[:, 0]  # Closing prices
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices
    daily_return_volatility = np.std(daily_returns)  # Volatility of returns
    features.append(daily_return_volatility)
    
    # Feature 2: Average Daily Volume (mean of the last 20 days)
    volumes = days[:, 4]  # Trading volumes
    avg_volume = np.mean(volumes)  # Average volume over the last 20 days
    features.append(avg_volume)

    # Feature 3: Short-Term Momentum (Close - 5-Day Average)
    short_term_avg = np.mean(closing_prices[-5:])  # 5-day moving average
    momentum = closing_prices[-1] - short_term_avg
    features.append(momentum)

    # Feature 4: Long-Term Trend (20-Day Moving Average)
    long_term_avg = np.mean(closing_prices[-20:])  # 20-day moving average
    trend = closing_prices[-1] - long_term_avg
    features.append(trend)
    
    # Feature 5: RSI with Historical Thresholds
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else np.nan
    rsi = 100 - (100 / (1 + rs)) if not np.isnan(rs) else 50  # Use 50 as a default
    
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0  # Initialize reward

    # Calculate historical thresholds for risk management
    risk_threshold_high = 0.7
    risk_threshold_moderate = 0.4
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        reward += -40  # Strong penalty for risky BUY
        # Mild positive reward for SELL-aligned features
        reward += 10  # Encouragement to sell
    elif risk_level > risk_threshold_moderate:
        # Moderate negative reward for BUY signals
        reward += -20  # Moderate penalty for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        if trend_direction > trend_threshold and features[2] > 0:  # Uptrend and positive momentum
            reward += 20  # Strong positive reward for correct direction
        elif trend_direction < -trend_threshold and features[2] < 0:  # Downtrend and negative momentum
            reward += 20  # Strong positive reward for correct direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < risk_threshold_moderate:
        if features[2] < 0:  # Assuming feature[2] indicates a negative momentum
            reward += 15  # Reward for mean-reversion actions
        else:
            reward -= 10  # Penalize for chasing trends

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward