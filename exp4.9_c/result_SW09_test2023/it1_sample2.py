import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: 5-day Moving Average
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    features.append(moving_average)

    # Feature 2: Price Momentum (current closing price - closing price 3 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-4] if len(closing_prices) >= 4 else 0
    features.append(price_momentum)

    # Feature 3: Volume Change (current volume - previous day's volume) / previous day's volume
    volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] if volumes[-2] != 0 else 1e-10)
    features.append(volume_change)

    # Feature 4: Volatility Ratio (current ATR vs historical mean ATR)
    atr = np.mean(np.abs(np.diff(closing_prices[-5:]))) if len(closing_prices) >= 5 else 0
    historical_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1e-10
    volatility_ratio = atr / historical_volatility
    features.append(volatility_ratio)

    # Feature 5: Crisis Signal (1 if volatility spikes beyond a threshold, else 0)
    crisis_signal = 1 if volatility_ratio > 2 else 0  # Example threshold
    features.append(crisis_signal)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for relative risk assessment
    risk_thresholds = [0.3, 0.7]
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_thresholds[1]:
        reward += -40  # Strong negative for BUY-aligned features
        reward += 5    # Mild positive for SELL-aligned features
    elif risk_level > risk_thresholds[0]:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_thresholds[0]:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Reward for upward momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_thresholds[0]:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_thresholds[0]:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return max(-100, min(100, reward))