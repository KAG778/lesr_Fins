import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Volumes

    # Feature 1: Recent Volatility (Standard deviation of daily returns over the last 20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    recent_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else np.std(daily_returns) if len(daily_returns) > 0 else 0
    features.append(recent_volatility)

    # Feature 2: Momentum (Price Change / Volume Change)
    if len(closing_prices) > 1:
        price_change = closing_prices[-1] - closing_prices[-2]
        volume_change = volumes[-1] - volumes[-2] + 1e-8  # To avoid division by zero
        momentum = price_change / volume_change
    else:
        momentum = 0.0
    features.append(momentum)

    # Feature 3: Average Daily Return over the last 20 days
    avg_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(avg_daily_return)

    # Feature 4: Maximum Drawdown over the last 30 days
    if len(closing_prices) >= 30:
        peak = np.maximum.accumulate(closing_prices[-30:])
        drawdown = (peak - closing_prices[-30:]) / peak
        max_drawdown = np.max(drawdown)
    else:
        max_drawdown = 0
    features.append(max_drawdown)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    avg_daily_return = features[2]
    recent_volatility = features[0]
    max_drawdown = features[3]
    momentum = features[1]

    reward = 0.0

    # Define relative thresholds for risk management based on historical data
    historical_std = np.std([avg_daily_return, recent_volatility, max_drawdown])
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 40  # Strong negative for BUY-aligned features
        if momentum < 0:  # Mild positive for SELL-aligned features
            reward += 10
        return np.clip(reward, -100, 100)

    if risk_level > medium_risk_threshold:
        reward -= 20  # Moderate negative for BUY-aligned features

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        if trend_direction > 0 and avg_daily_return > 0:  # Align with bullish momentum
            reward += 20
        elif trend_direction < 0 and avg_daily_return < 0:  # Align with bearish momentum
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if momentum < 0:  # Oversold condition
            reward += 15
        elif momentum > 0:  # Overbought condition
            reward -= 15

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clipping the reward to the range [-100, 100]
    return np.clip(reward, -100, 100)