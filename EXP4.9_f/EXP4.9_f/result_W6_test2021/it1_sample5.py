import numpy as np

def revise_state(s):
    # Extract relevant data
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: Price Momentum (Percentage Change from N days ago)
    N = 5  # Lookback period
    if len(closing_prices) >= N:
        price_momentum = (closing_prices[-1] - closing_prices[-N]) / closing_prices[-N]
    else:
        price_momentum = 0.0

    # Feature 2: Volatility (Standard deviation of closing prices over the last N days)
    if len(closing_prices) >= N:
        volatility = np.std(closing_prices[-N:])
    else:
        volatility = 0.0

    # Feature 3: Average Volume Change (Percentage change from the previous day)
    if len(volumes) > 1:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
    else:
        volume_change = 0.0

    # Feature 4: Exponential Moving Average (EMA) of closing prices (for trend detection)
    if len(closing_prices) >= N:
        ema = np.mean(closing_prices[-N:])  # Using mean as a simple approximation for EMA
    else:
        ema = 0.0

    # Feature 5: Relative Strength Index (RSI) over the last 14 days
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        gains = np.where(np.diff(prices) > 0, np.diff(prices), 0)
        losses = np.where(np.diff(prices) < 0, -np.diff(prices), 0)
        avg_gain = np.mean(gains[-period:]) if np.mean(gains[-period:]) != 0 else 0
        avg_loss = np.mean(losses[-period:]) if np.mean(losses[-period:]) != 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices[-14:])

    # Return features as a numpy array
    features = [price_momentum, volatility, volume_change, ema, rsi]
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
        reward += -40  # Strong negative reward for BUY-aligned features
        if features[0] < 0:  # If momentum is negative (suggesting sell)
            reward += 10  # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += 15
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += 15

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[4] > 70:  # Overbought condition, sell signal
            reward += 15
        elif features[4] < 30:  # Oversold condition, buy signal
            reward += 15

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds