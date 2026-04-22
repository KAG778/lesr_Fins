import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    features = []
    
    # Extract closing prices and daily returns
    closing_prices = s[::6]  # Extract closing prices
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

    # Feature 4: Moving Average Convergence Divergence (MACD)
    ema12 = np.mean(closing_prices[-12:])  # 12-day EMA
    ema26 = np.mean(closing_prices[-26:])  # 26-day EMA
    macd = ema12 - ema26
    features.append(macd)

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
    
    # Calculate thresholds based on historical data
    high_risk_threshold = 0.7  # This can be adjusted based on historical data
    low_risk_threshold = 0.4
    volatility_threshold = 0.6
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10    # Mild positive for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:
            reward += 15  # Reward for upward trend
        else:
            reward += 15  # Reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features
        reward -= 5   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds