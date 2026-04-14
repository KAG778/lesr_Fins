import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Initialize a list to store computed features
    features = []
    
    # Feature 1: 14-day Relative Strength Index (RSI)
    # Calculate the RSI based on closing prices
    closing_prices = s[0:120:6]  # Extracting closing prices
    price_changes = np.diff(closing_prices)
    
    # Avoid division by zero and handle NaN
    avg_gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
    avg_loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 2: Moving Average Convergence Divergence (MACD)
    # Calculate MACD using the last 20 closing prices
    short_ema = np.mean(closing_prices[-12:])  # Short-term EMA (12 days)
    long_ema = np.mean(closing_prices[-26:])   # Long-term EMA (26 days)
    macd = short_ema - long_ema
    features.append(macd)

    # Feature 3: Volatility (Standard deviation of returns)
    # Calculate the daily returns
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if returns.size > 0 else 0  # Avoid empty returns
    features.append(volatility)

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY-aligned features
        # Assume the feature indicates a SELL alignment
        reward += np.random.uniform(5, 10)  # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assume holding features are oversold/buy or overbought/sell
        reward += 5  # Reward mean-reversion features (oversold → buy, overbought → sell)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)