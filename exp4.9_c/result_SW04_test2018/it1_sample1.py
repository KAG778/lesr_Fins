import numpy as np

def revise_state(s):
    features = []
    
    # Calculate the daily returns
    closes = s[0::6]  # closing prices
    daily_returns = np.diff(closes) / closes[:-1]  # daily returns

    # Feature 1: Mean Daily Return
    mean_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
    features.append(mean_return)

    # Feature 2: Volatility (Standard Deviation of Returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
    features.append(volatility)

    # Feature 3: Relative Strength Index (RSI) for the last 14 days
    window_length = 14
    if len(daily_returns) > window_length:
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)

        avg_gain = np.mean(gains[-window_length:])
        avg_loss = np.mean(losses[-window_length:])

        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)
    else:
        features.append(50)  # Neutral RSI

    # Feature 4: Price Momentum (last close vs previous close)
    if len(closes) > 1:
        momentum = (closes[-1] - closes[-2]) / closes[-2]  # Normalize momentum
        features.append(momentum)
    else:
        features.append(0)

    # Feature 5: Price Range (High - Low of the last day)
    high_prices = s[2::6]
    low_prices = s[3::6]
    price_range = (high_prices[-1] - low_prices[-1]) / closes[-1] if closes[-1] != 0 else 0
    features.append(price_range)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(enhanced_s[123:])  # Use features from revise_state for volatility reference
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Strong penalty for buying in high risk
        # Mild positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10)
    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(5, 15)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for bullish features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3 * historical_volatility:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += 10  # Example positive reward for mean-reversion alignment

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Constrain reward within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward