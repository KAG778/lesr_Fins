import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:])  # 14-period RSI
    avg_loss = np.mean(loss[-14:])  # 14-period RSI
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))  # RSI calculation
    features.append(rsi)

    # Feature 2: Bollinger Bands (Upper and Lower Bands)
    moving_average = np.mean(closing_prices[-20:])  # 20-day moving average
    std_dev = np.std(closing_prices[-20:])           # 20-day standard deviation
    upper_band = moving_average + (2 * std_dev)
    lower_band = moving_average - (2 * std_dev)
    features.append(upper_band)
    features.append(lower_band)

    # Feature 3: Correlation with Market Index (Assuming market_index is provided)
    # Here we will simulate a market index correlation for the sake of the example
    market_index = np.random.normal(size=20)  # Simulated market index (replace with actual data)
    correlation = np.corrcoef(closing_prices[-20:], market_index)[0, 1]
    features.append(correlation)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Historical thresholds for determining relative risk levels
    historical_vol = np.std(enhanced_s[123:])  # Using features as a proxy for historical volatility
    high_risk_threshold = 0.7 * historical_vol
    medium_risk_threshold = 0.4 * historical_vol

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for buying
        reward += 20  # Mild positive for selling
    elif risk_level > medium_risk_threshold:
        reward -= np.random.uniform(5, 15)  # Moderate penalty for buying
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        reward += 20 * trend_direction  # Reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for potential mean-reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Halve the reward in high volatility

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits