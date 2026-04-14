import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    features = []

    # 1. Historical Volatility (standard deviation of returns over last 20 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Calculate daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
    features.append(historical_volatility)

    # 2. Bollinger Bands
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    upper_band = moving_average + 2 * historical_volatility
    lower_band = moving_average - 2 * historical_volatility
    features.append(upper_band)
    features.append(lower_band)

    # 3. Rate of Change (ROC) for momentum
    roc = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if closing_prices[-5] != 0 else 0
    features.append(roc)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 40  # Strong negative for BUY-aligned features
        reward += 5   # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive for upward features
        else:  # Downtrend
            reward += 20  # Positive for downward features

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Check if the price is near the lower Bollinger Band for potential buy
        if enhanced_s[121] < enhanced_s[122]:  # Lower band < current price
            reward += 15  # Reward mean-reversion buy signal
        # Check if the price is near the upper Bollinger Band for potential sell
        elif enhanced_s[121] > enhanced_s[122]:  # Upper band > current price
            reward += 15  # Reward mean-reversion sell signal

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds