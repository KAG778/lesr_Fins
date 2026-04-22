import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes
    
    # Feature 1: Price Change Percentage from the last two days
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Average Trading Volume (last 10 days)
    average_volume = np.mean(trading_volumes[-10:]) if len(trading_volumes) >= 10 else 0.0
    
    # Feature 3: Price Momentum (Change from two days ago to the last day)
    price_momentum = closing_prices[-1] - closing_prices[-2]
    
    # Feature 4: Historical Volatility (Standard deviation of the last 20 closing prices)
    historical_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    
    # Feature 5: Momentum Z-score (Z-score of the last day's price change)
    recent_price_changes = np.diff(closing_prices[-10:])  # Last 10 days price changes
    mean_price_change = np.mean(recent_price_changes)
    std_price_change = np.std(recent_price_changes) if np.std(recent_price_changes) != 0 else 1
    z_score_price_change = (price_change_pct - mean_price_change) / std_price_change

    features = [price_change_pct, average_volume, price_momentum, historical_volatility, z_score_price_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    avg_risk = np.mean(features)
    std_risk = np.std(features)
    risk_threshold_high = avg_risk + 2 * std_risk
    risk_threshold_mid = avg_risk + std_risk

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if features[0] > 0:  # Positive price change indicates a BUY signal
            reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        else:
            reward += np.random.uniform(5, 10)    # Mild positive for SELL
    elif risk_level > risk_threshold_mid:
        if features[0] > 0:  # Positive price change indicates a BUY signal
            reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_mid:
        if trend_direction > 0:  # Uptrend
            if features[2] > 0:  # Price momentum aligns
                reward += np.random.uniform(10, 20)
        elif trend_direction < 0:  # Downtrend
            if features[2] < 0:  # Price momentum aligns
                reward += np.random.uniform(10, 20)

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Strongly oversold condition
            reward += np.random.uniform(10, 20)  # Reward for buying
        elif features[0] > 0.05:  # Strongly overbought condition
            reward += np.random.uniform(10, 20)  # Reward for selling

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]