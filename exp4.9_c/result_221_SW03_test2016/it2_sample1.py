import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    # Feature 1: Price Momentum (closing price difference with the previous day)
    momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) >= 2 else 0.0

    # Feature 2: Average Daily Volume (last 5 trading days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1e-10  # Prevent div by zero

    # Feature 3: Z-Score of Price (mean reversion indicator)
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:])
        z_score = (closing_prices[-1] - mean_price) / std_price if std_price > 0 else 0.0
    else:
        z_score = 0.0

    # Feature 4: Combined Volatility and Momentum Indicator
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0.0
    vol_momentum = momentum / (volatility + 1e-10)  # Normalize by volatility to avoid division by zero

    # Collect features
    features = [momentum, avg_volume, z_score, vol_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds based on recent data
    historical_std = np.std(enhanced_s[123:])  # Use the std of features for relative thresholds
    risk_thresholds = {
        'low': 0.4 * historical_std,
        'medium': 0.7 * historical_std,
        'high': 1.0 * historical_std
    }

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_thresholds['high']:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10   # Mild positive reward for SELL-aligned features
    elif risk_level > risk_thresholds['medium']:
        reward -= 20  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < risk_thresholds['medium']:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_thresholds['low']:
        z_score = enhanced_s[123]  # Assuming z_score is the third feature in revised state
        if z_score < -1:  # Oversold condition
            reward += 15  # Reward for potential buy
        elif z_score > 1:  # Overbought condition
            reward += 15  # Reward for potential sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return float(np.clip(reward, -100, 100))