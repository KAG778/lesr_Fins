import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    trading_volumes = s[4::6]  # Extracting trading volumes
    
    # Feature 1: Volatility-adjusted Price Momentum
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / np.std(closing_prices[-20:]) if len(closing_prices) > 1 else 0.0
    
    # Feature 2: Z-score of Volume
    avg_volume = np.mean(trading_volumes[-20:]) if len(trading_volumes) > 20 else 0.0
    std_volume = np.std(trading_volumes[-20:]) if len(trading_volumes) > 20 else 1.0  # Use 1.0 to avoid division by zero
    volume_z_score = (trading_volumes[-1] - avg_volume) / std_volume if avg_volume != 0 else 0.0
    
    # Feature 3: Bollinger Bands - Latest price relative to the upper and lower bands
    moving_avg = np.mean(closing_prices[-20:]) if len(closing_prices) > 20 else 0.0
    upper_band = moving_avg + 2 * np.std(closing_prices[-20:]) if len(closing_prices) > 20 else moving_avg
    lower_band = moving_avg - 2 * np.std(closing_prices[-20:]) if len(closing_prices) > 20 else moving_avg
    price_position_in_bands = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.0
    
    features = [price_momentum, volume_z_score, price_position_in_bands]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Newly computed features
    reward = 0.0
    
    # Define historical thresholds based on mean and std dev of features
    risk_threshold = 0.7  # Example, could be replaced with dynamic threshold
    trend_threshold = 0.3  # Example, could be replaced with dynamic threshold

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold:
        # Strong negative reward for BUY-aligned features
        reward -= 40 if features[0] > 0 else 0  # Price momentum positive
        # Mild positive reward for SELL-aligned features
        reward += 10 if features[0] < 0 else 0  # Price momentum negative

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold and features[0] > 0:  # Uptrend and positive momentum
            reward += 20
        elif trend_direction < -trend_threshold and features[0] < 0:  # Downtrend and negative momentum
            reward += 20

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[2] < 0:  # Assuming price is below the lower Bollinger Band
            reward += 15  # Reward for buying in oversold conditions
        elif features[2] > 1:  # Assuming price is above the upper Bollinger Band
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range