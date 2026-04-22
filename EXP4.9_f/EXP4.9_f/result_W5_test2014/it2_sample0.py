import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Change Percentage from the last day
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Volume Change Percentage from the last day
    volume_change_pct = (trading_volumes[-1] - trading_volumes[-2]) / trading_volumes[-2] if trading_volumes[-2] != 0 else 0.0
    
    # Feature 3: Historical Volatility (standard deviation of the last 20 days)
    historical_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    
    # Feature 4: Z-score of Recent Price Change Percentage (last 5 days)
    recent_price_changes = np.diff(closing_prices[-5:])
    mean_recent_change = np.mean(recent_price_changes)
    std_recent_change = np.std(recent_price_changes) if np.std(recent_price_changes) != 0 else 1
    z_score_recent_change = (price_change_pct - mean_recent_change) / std_recent_change

    # Feature 5: Rate of Change (ROC) over last 10 days
    roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) > 10 else 0.0

    features = [price_change_pct, volume_change_pct, historical_volatility, z_score_recent_change, roc]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate relative thresholds based on historical data from features
    historical_std = np.std(features)  # Using standard deviation of the features for risk thresholding
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:  # High-risk condition
        reward -= np.random.uniform(30, 50) if features[0] > 0 else 0  # Strong negative reward for positive price change
        reward += np.random.uniform(5, 10) if features[0] < 0 else 0  # Mild positive reward for negative price change
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20) if features[0] > 0 else 0  # Moderate negative reward for positive price change

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if (trend_direction > 0 and features[0] > 0) or (trend_direction < 0 and features[0] < 0):
            reward += np.random.uniform(10, 20)  # Reward for alignment of trend and price momentum

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Strongly oversold condition
            reward += np.random.uniform(10, 20)  # Reward for buying in mean reversion
        elif features[0] > 0.05:  # Strongly overbought condition
            reward += np.random.uniform(10, 20)  # Reward for selling in mean reversion

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]