import numpy as np

def revise_state(s):
    # Extract closing prices and volumes from the raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    features = []

    # Feature 1: Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else np.mean(gain)
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else np.mean(loss)
        
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices)
    features.append(rsi)  # Append RSI as a feature

    # Feature 2: Price Volatility (Standard Deviation of Returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0  # Last 5 days
    features.append(volatility)  # Append volatility as a feature

    # Feature 3: Bollinger Bands (Upper Band - Lower Band)
    if len(closing_prices) >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        bollinger_band_width = upper_band - lower_band
    else:
        bollinger_band_width = 0  # Not enough data
    features.append(bollinger_band_width)  # Append Bollinger Bands width

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for thresholds relative to std dev
    historical_data = enhanced_s[0:120]  # Assuming we have historical data
    historical_returns = np.diff(historical_data[0::6]) / historical_data[0::6][:-1]
    mean_return = np.mean(historical_returns)
    std_return = np.std(historical_returns)

    # Initialize reward
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50  # Strong negative for BUY when risk is high
        reward += 10 if trend_direction < 0 else 0  # Mild positive for SELL when risk is high
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY when risk is moderate

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 30  # Strong positive for upward momentum alignment
        elif trend_direction < -0.3:
            reward += 30  # Strong positive for downward momentum alignment

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3:
        if risk_level < 0.3:
            reward += 15  # Reward mean-reversion features
        else:
            reward -= 10  # Penalty for trading in sideways markets with high risk

    # Priority 4: High Volatility
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Normalize reward to be within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward