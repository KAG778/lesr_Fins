import numpy as np

def revise_state(s):
    features = []

    # Extract relevant price data
    closing_prices = s[0::6]  # Closing prices of the last 20 days
    high_prices = s[2::6]      # High prices of the last 20 days
    low_prices = s[3::6]       # Low prices of the last 20 days
    volumes = s[4::6]          # Trading volumes of the last 20 days

    # Feature 1: Z-score of daily returns (to capture momentum)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    z_score = (np.mean(daily_returns) - np.mean(daily_returns[-14:])) / np.std(daily_returns[-14:]) if len(daily_returns) > 14 else 0
    features.append(z_score)

    # Feature 2: Historical volatility (standard deviation of returns)
    volatility = np.std(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    features.append(volatility)

    # Feature 3: Rate of Change (ROC) of closing prices over the last 10 days
    roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) > 10 else 0
    features.append(roc)

    # Feature 4: Average Directional Index (ADX) for trend strength
    adx = np.mean(np.abs(np.diff(closing_prices[-14:]))) if len(closing_prices) >= 14 else 0
    features.append(adx)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for risk management based on past features
    historical_volatility = np.std(enhanced_s[123:])  # Use features for volatility context
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.4 * historical_volatility

    # Initialize the reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for risky buy signals
        reward += np.random.uniform(5, 10)    # Mild positive for sell signals
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(5, 15)  # Moderate negative for buy signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 20 * np.sign(trend_direction)  # Positive reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 10  # Reward for mean-reversion alignment

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_volatility and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Constrain reward within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return float(reward)