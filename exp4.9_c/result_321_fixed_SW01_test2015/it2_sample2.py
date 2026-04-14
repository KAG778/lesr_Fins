import numpy as np

def revise_state(s):
    # Extract closing prices
    closing_prices = s[0:120:6]  # Closing prices
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

    rsi = calculate_rsi(closing_prices) if num_days >= 14 else 0.0

    # Feature 2: Average True Range (ATR) for volatility
    high_prices = s[3:120:6]
    low_prices = s[2:120:6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                         np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0

    # Feature 3: 5-Day Volatility (standard deviation of the last 5 closes)
    volatility = np.std(closing_prices[-5:]) if num_days >= 5 else 0.0

    # Feature 4: Price Change Percentage (from previous day)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if num_days > 1 else 0.0

    # Feature 5: Cumulative Return over the last 10 days
    cumulative_return = (closing_prices[-1] / closing_prices[-11] - 1) * 100 if num_days >= 10 else 0.0

    features = [rsi, atr, volatility, price_change_pct, cumulative_return]
    
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
        reward -= 50.0  # Strong negative for BUY in high risk
        reward += 5.0 * abs(features[0])  # Mild positive for SELL-aligned features based on RSI
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Align reward with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 30:  # RSI < 30: Oversold condition
            reward += 15.0  # Positive for buying in oversold condition
        elif features[0] > 70:  # RSI > 70: Overbought condition
            reward -= 15.0  # Negative for selling in overbought condition

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))