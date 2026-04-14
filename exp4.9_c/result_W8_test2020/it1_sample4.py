import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    
    # 1. 20-day moving average
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    features.append(moving_average)
    
    # 2. Price momentum (current price minus price from N days ago)
    N = 5
    price_momentum = closing_prices[-1] - closing_prices[-N] if len(closing_prices) >= N else 0.0
    features.append(price_momentum)
    
    # 3. Volatility over the last 20 days (standard deviation)
    volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    features.append(volatility)
    
    # 4. Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 and closing_prices[-2] != 0 else 0.0
    features.append(roc)
    
    # 5. Bollinger Bands: Upper and Lower Bands for mean reversion signals
    rolling_mean = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    rolling_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    features.append(upper_band)
    features.append(lower_band)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2: Trend Following when risk is low
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Positive reward for upward trend
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Positive reward for downward trend

    # Priority 3: Mean Reversion during sideways markets
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        oversold_threshold = enhanced_s[123] < (enhanced_s[4] if enhanced_s[4] > 0 else 0)  # Lower band
        overbought_threshold = enhanced_s[123] > (enhanced_s[5] if enhanced_s[5] > 0 else 0)  # Upper band
        if oversold_threshold:
            reward += 10  # Reward for buying an oversold signal
        elif overbought_threshold:
            reward += 10  # Reward for selling an overbought signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clamp the reward between -100 and 100