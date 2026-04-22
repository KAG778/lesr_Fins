import numpy as np

def revise_state(s):
    # Features list
    features = []
    
    # Extract closing prices and calculate daily returns
    closing_prices = s[0::6]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    daily_returns = np.nan_to_num(daily_returns)  # Replace NaNs with 0
    
    # Feature 1: Average Daily Return
    avg_daily_return = np.mean(daily_returns)

    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)
    
    # Feature 3: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 4: Relative Strength Index (RSI) - 14-day
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) > 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) > 14 else 0
    rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))  # Avoid division by zero

    # Feature 5: Mean Reversion Indicator (using Bollinger Bands)
    rolling_mean = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    rolling_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1  # Avoid division by zero
    mean_reversion_indicator = (closing_prices[-1] - rolling_mean) / rolling_std  # Z-score

    features.extend([avg_daily_return, volatility, price_momentum, rsi, mean_reversion_indicator])
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Calculate thresholds based on historical data (for example, using standard deviations)
    historical_volatility = 0.2  # This should be dynamically calculated based on historical data
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.4 * historical_volatility
    trend_threshold = 0.3
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
        reward += 10   # Mild positive for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < low_risk_threshold:
        if trend_direction > trend_threshold:
            reward += 20  # Reward for bullish trends
        elif trend_direction < -trend_threshold:
            reward += 20  # Reward for bearish trends

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds