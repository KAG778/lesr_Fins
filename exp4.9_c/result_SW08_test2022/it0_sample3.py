import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (day i at index i*6)
    trading_volumes = s[4:120:6]  # Extract trading volumes

    # Feature 1: Daily Return
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily return as percentage
    daily_returns = np.insert(daily_returns, 0, 0)  # Insert 0 for the first day

    # Feature 2: Moving Average (5-day moving average)
    moving_avg = np.convolve(closing_prices, np.ones(5)/5, mode='valid')
    moving_avg = np.concatenate((np.full(4, np.nan), moving_avg))  # Fill the beginning with NaN

    # Feature 3: Volume Change
    volume_change = np.diff(trading_volumes) / trading_volumes[:-1]  # Volume change as percentage
    volume_change = np.insert(volume_change, 0, 0)  # Insert 0 for the first day

    # Creating the feature array
    features = []
    features.append(np.mean(daily_returns))  # Average daily return
    features.append(np.nanmean(moving_avg))  # Average of moving average
    features.append(np.mean(volume_change))  # Average volume change

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
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        if features[0] > 0:  # assuming feature[0] relates to buying signals
            reward = -40.0  # Strong penalty for buying
        # MILD POSITIVE reward for SELL-aligned features
        if features[0] < 0:  # assuming feature[0] relates to selling signals
            reward = 10.0  # Mild reward for selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # assuming feature[0] relates to buying signals
            reward = -15.0  # Moderate penalty for buying

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and favorable features
            reward = 20.0  # Positive reward for buying signals in uptrend
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and favorable features
            reward = 20.0  # Positive reward for selling signals in downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Assuming oversold signal for buying
            reward = 15.0  # Reward for buying oversold
        elif features[0] > 0:  # Assuming overbought signal for selling
            reward = -10.0  # Penalty for buying overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)