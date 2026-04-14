import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    num_days = len(closing_prices)

    # Feature 1: Price Momentum (current close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2] if num_days > 1 else 0.0

    # Feature 2: Average True Range (ATR) for volatility
    high_prices = s[3:120:6]    # Extract high prices
    low_prices = s[2:120:6]      # Extract low prices
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                         np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0  # 14-day ATR

    # Feature 3: 14-Day Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices) if num_days >= 14 else 0.0

    # Feature 4: Standard Deviation of Returns (Risk Indicator)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if len(returns) > 0 else 0.0

    # Feature 5: Moving Average Convergence Divergence (MACD)
    if num_days >= 26:
        short_ema = np.mean(closing_prices[-12:])  # Short-term EMA (12 days)
        long_ema = np.mean(closing_prices[-26:])   # Long-term EMA (26 days)
        macd = short_ema - long_ema
    else:
        macd = 0.0

    features = [price_momentum, atr, rsi, volatility, macd]
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
        reward -= 40.0  # Strong negative for BUY in high risk
        reward += 5.0 * abs(features[0])  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Reward for momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # RSI < 30: Oversold
            reward += 10.0  # Positive for buying in oversold condition
        elif features[2] > 70:  # RSI > 70: Overbought
            reward -= 10.0  # Negative for selling in overbought condition

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))