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
    
    # Feature 4: Price Momentum (change in price from last 5 days)
    price_momentum = np.mean(np.diff(closing_prices[-5:])) if len(closing_prices) >= 6 else 0.0
    
    # Feature 5: Volume Momentum (change in volume from last 5 days)
    volume_momentum = np.mean(np.diff(trading_volumes[-5:])) if len(trading_volumes) >= 6 else 0.0
    
    features = [price_change_pct, volume_change_pct, historical_volatility, price_momentum, volume_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract computed features
    reward = 0.0

    # Calculate relative thresholds
    price_change_pct_std = np.std(features[0]) if len(features) > 0 else 1
    volume_change_pct_std = np.std(features[1]) if len(features) > 1 else 1
    risk_threshold_high = 0.7  # High risk threshold
    risk_threshold_mid = 0.4    # Medium risk threshold
    trend_threshold = 0.3        # Trend threshold

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Positive price change indicates a BUY signal
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        else:
            reward += np.random.uniform(5, 10)

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_mid:
        if trend_direction > trend_threshold and features[3] > 0:  # Uptrend with positive price momentum
            reward += np.random.uniform(10, 20)
        elif trend_direction < -trend_threshold and features[3] < 0:  # Downtrend with negative price momentum
            reward += np.random.uniform(10, 20)

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[0] < -0.05:  # Strongly oversold condition
            reward += np.random.uniform(10, 20)
        elif features[0] > 0.05:  # Strongly overbought condition
            reward += np.random.uniform(10, 20)

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]