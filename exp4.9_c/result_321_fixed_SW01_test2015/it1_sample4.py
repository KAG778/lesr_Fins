import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    num_days = len(closing_prices)
    
    # Feature 1: 14-Day Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices)

    # Feature 2: 10-Day Price Volatility (Standard Deviation)
    price_volatility = np.std(closing_prices[-10:]) if num_days >= 10 else 0.0

    # Feature 3: Price Change over Last 5 Days (Percentage)
    price_change_5d = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100 if num_days >= 6 else 0.0

    # Feature 4: Price Momentum (Current Close - Previous Close)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0.0
    
    features = [rsi, price_volatility, price_change_5d, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50.0  # Strong negative for BUY-aligned features
        reward += 5.0 * abs(features[3])  # Mild positive for SELL-aligned features based on price momentum
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[3] * 15.0  # Strong positive reward for upward momentum
        else:  # Downtrend
            reward -= features[3] * 15.0  # Strong positive reward for downward momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 30:  # RSI < 30: Oversold condition
            reward += 15.0  # Positive for buying in oversold condition
        elif features[0] > 70:  # RSI > 70: Overbought condition
            reward -= 15.0  # Positive for selling in overbought condition

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))