import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    features = []

    # 1. 20-day Historical Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Calculate daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
    features.append(historical_volatility)

    # 2. Crisis Indicator (percentage drop from peak in the last 20 days)
    peak_price = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    current_price = closing_prices[-1]
    crisis_indicator = ((peak_price - current_price) / peak_price) * 100  # Percentage drop
    features.append(crisis_indicator)

    # 3. Volume Spike (Relative to 20-day average)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    volume_spike = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_spike)

    # 4. Rate of Change (ROC) for price momentum over the last 5 days
    if len(closing_prices) >= 6:
        roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    else:
        roc = 0
    features.append(roc)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_std = np.std(enhanced_s[123:])  # Use features for dynamic thresholds
    risk_threshold_high = historical_std * 1.5  # High risk threshold
    risk_threshold_medium = historical_std * 0.75  # Medium risk threshold

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10 if trend_direction < 0 else -5  # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += 25  # Strong positive for upward momentum
        else:  # Downtrend
            reward += 15  # Moderate positive for downward momentum

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3:
        if risk_level < 0.3:  # Low risk
            reward += 20  # Reward mean-reversion features
        else:
            reward += -10  # Penalize for chasing breakouts

    # Priority 4: HIGH VOLATILITY
    if volatility_level > historical_std * 1.5:  # Example condition for high volatility
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds