import numpy as np

def revise_state(s):
    # s: 120d raw state
    # We will compute three features:
    # 1. Momentum (Rate of Change)
    # 2. Volatility (Standard Deviation of the last 5 closing prices)
    # 3. Moving Average Convergence Divergence (MACD)

    closing_prices = s[0::6]  # Extract closing prices
    if len(closing_prices) < 20:
        return np.array([])  # Handle edge case

    # Feature 1: Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Feature 2: Volatility (Standard Deviation of the last 5 closing prices)
    volatility = np.std(closing_prices[-5:]) if len(closing_prices[-5:]) > 1 else 0

    # Feature 3: MACD
    # Simple MACD calculation using the last 12 and 26 periods for EMA
    ema_short = np.mean(closing_prices[-12:]) if len(closing_prices[-12:]) >= 12 else 0
    ema_long = np.mean(closing_prices[-26:]) if len(closing_prices[-26:]) >= 26 else 0
    macd = ema_short - ema_long

    return np.array([roc, volatility, macd])

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if enhanced_s[123] > 0:  # Assuming positive feature indicates a strong BUY signal
            return np.random.uniform(-50, -30)  # Strong negative reward for BUY
        elif enhanced_s[123] < 0:  # Assuming negative feature indicates a strong SELL signal
            return np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > 0.4:
        if enhanced_s[123] > 0:
            reward += np.random.uniform(-20, -10)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] > 0:  # Positive trend and BUY signal
            reward += np.random.uniform(10, 20)  # Positive reward for aligning with trend
        elif trend_direction < -0.3 and enhanced_s[123] < 0:  # Negative trend and SELL signal
            reward += np.random.uniform(10, 20)  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] > 0:  # Oversold condition for BUY
            reward += np.random.uniform(10, 20)  # Reward for mean-reversion buy
        elif enhanced_s[123] < 0:  # Overbought condition for SELL
            reward += np.random.uniform(10, 20)  # Reward for mean-reversion sell

    # Priority 4 — HIGH VOLATILITY (no crisis)
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)