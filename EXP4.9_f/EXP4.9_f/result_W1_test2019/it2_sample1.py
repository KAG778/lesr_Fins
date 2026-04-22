import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and daily returns
    closing_prices = s[0::6]  # Closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    daily_returns = np.append(np.nan, daily_returns)  # Handle NaN for the first day
    daily_returns = np.nan_to_num(daily_returns)  # Replace NaN with 0

    # Feature 1: Average Daily Return
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)

    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    # Feature 3: Price Momentum (latest close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    features.append(price_momentum)

    # Feature 4: Drawdown from the peak in the last 20 days
    peak = np.max(closing_prices[-20:]) if len(closing_prices) > 20 else closing_prices[-1]
    drawdown = (peak - closing_prices[-1]) / peak if peak != 0 else 0
    features.append(drawdown)

    # Feature 5: Average True Range (ATR) for volatility measurement
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds for risk management
    risk_thresholds = np.percentile([0.2, 0.4, 0.6, 0.8], [0.25, 0.5, 0.75, 1.0])  # Example thresholds based on historical data

    # Initialize the reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_thresholds[2]:  # High risk
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10    # Mild positive for SELL-aligned features
    elif risk_level > risk_thresholds[1]:  # Moderate risk
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_thresholds[1]:  # Low risk and strong trend
        reward += 15 if trend_direction > 0 else 10  # Positive reward for alignment with momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_thresholds[0]:  # Low risk and sideways
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_thresholds[1]:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds