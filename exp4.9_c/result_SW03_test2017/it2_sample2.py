import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]         # Trading volumes

    # Feature 1: Exponential Moving Average (EMA) - 20 days
    def ema(prices, span=20):
        if len(prices) < span:
            return 0
        return np.mean(prices[-span:])  # Simplified EMA for demonstration
    
    features.append(ema(closing_prices))

    # Feature 2: Average True Range (ATR) - 14 days
    def atr(highs, lows, closes, period=14):
        if len(highs) < period:
            return 0
        tr = np.maximum(highs[-period:] - lows[-period:], 
                        np.maximum(np.abs(highs[-period:] - closes[-period-1:-1]), 
                                   np.abs(lows[-period:] - closes[-period-1:-1])))
        return np.mean(tr)

    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    features.append(atr(high_prices, low_prices, closing_prices))

    # Feature 3: Z-score of daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    if len(daily_returns) > 0:
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        z_score = (daily_returns[-1] - mean_return) / std_return if std_return > 0 else 0
        features.append(z_score)
    else:
        features.append(0)

    # Feature 4: Current Price Relative to 20-day Moving Average
    moving_avg = ema(closing_prices, span=20)
    current_price = closing_prices[-1]
    price_relative_to_ma = (current_price - moving_avg) / moving_avg if moving_avg != 0 else 0
    features.append(price_relative_to_ma)

    # Feature 5: Rate of Change (momentum) over the last 5 days
    if len(closing_prices) > 5 and closing_prices[-6] != 0:
        momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
    else:
        momentum = 0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical std deviation for relative thresholds
    historical_features = enhanced_s[123:]
    historical_std = np.std(historical_features)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50 * (risk_level - 0.7)  # Strong negative for risky BUY signals
        if trend_direction < 0:  # Mild positive for SELL-aligned features
            reward += 10 * (1 - risk_level)
    elif risk_level > 0.4:
        reward -= 20 * (risk_level - 0.4)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 20 * abs(trend_direction)  # Reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std * 1.5:  # Relative measure
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within specified range