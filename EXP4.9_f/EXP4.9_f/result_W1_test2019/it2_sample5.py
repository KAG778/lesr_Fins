import numpy as np

def revise_state(s):
    # Extract closing prices, high prices, and low prices
    closing_prices = s[::6]  # Every 6th element starting from index 0
    high_prices = s[2::6]     # Every 6th element starting from index 2
    low_prices = s[3::6]      # Every 6th element starting from index 3

    # Ensure we have enough data for computations
    if len(closing_prices) < 20:
        return np.zeros(5)  # Return zero features if not enough data

    # Feature 1: Average Daily Return
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0

    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)

    # Feature 3: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 4: Drawdown from the peak in the last 20 days
    peak = np.max(closing_prices[-20:])
    drawdown = (peak - closing_prices[-1]) / peak if peak != 0 else 0

    # Feature 5: Average True Range (ATR) for volatility measurement
    true_range = np.maximum(high_prices[1:] - low_prices[1:],
                            np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                                       np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_range[-14:]) if len(true_range) >= 14 else 0  # 14-day ATR

    return np.array([avg_daily_return, volatility, price_momentum, drawdown, atr])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Relative thresholds based on historical standard deviations
    risk_thresholds = np.array([0.2, 0.4, 0.6, 0.8])  # Example thresholds
    risk_boundaries = np.percentile(risk_level, risk_thresholds)

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_boundaries[2]:  # High risk
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10    # Mild positive for SELL-aligned features
    elif risk_level > risk_boundaries[1]:  # Moderate risk
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_boundaries[1]:  # Low risk and strong trend
        if trend_direction > 0:
            reward += 15  # Reward for upward trend
        else:
            reward += 15  # Reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_boundaries[0]:  # Low risk and sideways
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_boundaries[1]:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within bounds