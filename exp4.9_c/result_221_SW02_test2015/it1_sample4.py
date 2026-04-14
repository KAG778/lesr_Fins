import numpy as np

def revise_state(s):
    features = []
    
    # Extract relevant data from the state
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    # Ensure at least 20 days of data for calculations
    if len(closing_prices) >= 20:
        # Feature 1: 20-day moving average
        moving_average = np.mean(closing_prices[-20:])
        features.append(moving_average)

        # Feature 2: Bollinger Bands
        rolling_std = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * rolling_std)
        lower_band = moving_average - (2 * rolling_std)
        current_price = closing_prices[-1]
        features.append(1 if current_price > upper_band else -1 if current_price < lower_band else 0)

        # Feature 3: Average True Range (ATR) for volatility
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                   abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
        features.append(atr)

        # Feature 4: Price Momentum (percentage change)
        price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
        features.append(price_momentum)

        # Feature 5: Volume Change (percentage change)
        if len(volumes) >= 2:
            volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
            features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_std = np.std(enhanced_s[123:]) if len(enhanced_s[123:]) > 0 else 1  # Prevent division by zero
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY
        reward += np.random.uniform(5, 10)   # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        features = enhanced_s[123:]  # Extract new features
        if trend_direction > 0 and features[3] > 0:  # Assuming features[3] indicates upward price momentum
            reward += 15  # Positive reward for aligning with the trend
        elif trend_direction < 0 and features[3] < 0:  # Assuming features[3] indicates downward price momentum
            reward += 15  # Positive reward for aligning with the trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        features = enhanced_s[123:]  # Extract new features
        if features[1] == -1:  # Assuming -1 indicates oversold condition
            reward += 15  # Reward for buying in an oversold condition
        elif features[1] == 1:  # Assuming 1 indicates overbought condition
            reward -= 15  # Penalize for buying in an overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds