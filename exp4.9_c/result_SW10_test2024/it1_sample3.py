import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volume = s[4:120:6]  # Extract trading volumes
    features = []

    # 1. Price Change Rate (Percentage Change over the last 5 days)
    if len(closing_prices) >= 6:
        price_change_rate = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
    else:
        price_change_rate = 0  # Not enough data
    features.append(price_change_rate)

    # 2. Volume Spike (Relative to 20-day average)
    avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
    volume_spike = (volume[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_spike)

    # 3. Moving Average Convergence Divergence (MACD)
    if len(closing_prices) >= 26:
        short_ema = np.mean(closing_prices[-12:])  # 12-day EMA
        long_ema = np.mean(closing_prices[-26:])   # 26-day EMA
        macd = short_ema - long_ema
    else:
        macd = 0  # Not enough data
    features.append(macd)

    # 4. Historical Volatility (Standard Deviation of returns over the last 20 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
    features.append(historical_volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE for BUY-aligned features
        reward += 10   # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 15 * np.sign(trend_direction)  # Align reward with trend direction

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features
        reward -= 5   # Penalize breakout-chasing features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds