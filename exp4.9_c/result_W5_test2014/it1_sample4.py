import numpy as np

def revise_state(s):
    # Extract closing prices and volumes from the state
    closing_prices = s[0::6]  # every 6th element starting from index 0
    volumes = s[4::6]         # every 6th element starting from index 4

    # Feature 1: Price Momentum (current close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0

    # Feature 2: Relative Strength Index (RSI) over the last 14 days
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices)

    # Feature 3: Moving Average Convergence Divergence (MACD)
    def calculate_macd(prices):
        short_ema = np.mean(prices[-12:]) if len(prices) >= 12 else 0
        long_ema = np.mean(prices[-26:]) if len(prices) >= 26 else 0
        return short_ema - long_ema

    macd = calculate_macd(closing_prices)

    # Feature 4: Volatility (Standard Deviation of returns over the last 20 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0  # Last 20 days

    # Feature 5: Average Volume (last 20 days)
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0

    features = [price_momentum, rsi, macd, volatility, average_volume]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    
    # Calculate relative thresholds based on historical standard deviations
    recent_risk = np.std(features[0:5])  # Use volatility feature for risk assessment
    recent_rsi = features[1]
    
    # Setting initial reward
    reward = 0.0

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming feature[0] relates to BUY signal
            reward += -50  # Strong negative reward for buying in high risk
        # Mild positive reward for SELL-aligned features
        elif features[0] < 0:  # Assuming feature[0] relates to SELL signal
            reward += 10  # Mild positive reward for selling in high risk

    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Assuming feature[0] relates to BUY signal
            reward += -20  # Moderate negative reward for buying in elevated risk

    # **Priority 2 — TREND FOLLOWING**
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Assuming feature[0] relates to upward signal
            reward += 20  # Strong positive reward for correct bullish signal
        elif trend_direction < -0.3 and features[0] < 0:  # Assuming feature[0] relates to downward signal
            reward += 20  # Strong positive reward for correct bearish signal

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if recent_rsi < 30:  # Oversold condition
            reward += 15  # Reward for buying in oversold conditions
        elif recent_rsi > 70:  # Overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return float(np.clip(reward, -100, 100))