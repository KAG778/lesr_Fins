import numpy as np

def revise_state(s):
    features = []
    
    # Extract relevant price and volume data
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes
    
    # Feature 1: Average Daily Return
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    daily_returns = np.append(np.nan, daily_returns)  # Handle NaN for the first entry
    avg_daily_return = np.nanmean(daily_returns)  # Calculate mean ignoring NaNs
    features.append(avg_daily_return)

    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)  # Standard deviation of daily returns
    features.append(volatility)

    # Feature 3: Price Momentum (latest close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    features.append(price_momentum)

    # Feature 4: Drawdown from Peak (last 20 days)
    peak = np.max(closing_prices[-20:]) if len(closing_prices) > 20 else closing_prices[-1]
    drawdown = (peak - closing_prices[-1]) / peak if peak != 0 else 0
    features.append(drawdown)

    # Feature 5: Relative Strength Index (RSI) - 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.nanmean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.nanmean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Relative thresholds based on historical data (example values)
    risk_thresholds = np.array([0.1, 0.3, 0.5, 0.7])  # Example thresholds for various risk levels
    volatility_threshold = np.std(risk_level)  # Use historical std for volatility levels

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_thresholds[3]:  # High risk
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10    # Mild positive for SELL-aligned features
    elif risk_level > risk_thresholds[2]:  # Moderate risk
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_thresholds[2]:  # Low risk and strong trend
        if trend_direction > 0:
            reward += 20  # Positive reward for upward trend
        else:
            reward += 20  # Positive reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_thresholds[1]:  # Low risk and sideways
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < risk_thresholds[2]:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds