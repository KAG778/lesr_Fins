import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    opening_prices = s[1::6]  # Extracting opening prices
    high_prices = s[2::6]     # Extracting high prices
    low_prices = s[3::6]      # Extracting low prices
    volumes = s[4::6]         # Extracting volumes
    
    # Feature 1: Average Daily Return
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0.0
    
    # Feature 2: Volatility Measure (using historical standard deviation of daily returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0.0
    
    # Feature 3: Z-Score of Current Price Relative to Moving Average (20-day MA)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    z_score = (closing_prices[-1] - moving_average) / volatility if volatility != 0 else 0.0

    features = [avg_daily_return, volatility, z_score]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY in high-risk situations
        reward += 10.0 if features[0] < 0 else 0  # Mild positive for SELL if average return is negative
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 30.0  # Strong reward for momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Z-Score indicates oversold
            reward += 15.0  # Strong buy signal
        elif features[2] > 1:  # Z-Score indicates overbought
            reward += -15.0  # Strong sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high-volatility conditions

    return float(np.clip(reward, -100, 100))