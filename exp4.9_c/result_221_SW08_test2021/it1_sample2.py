import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) for trend detection
    def compute_ema(prices, period=10):
        if len(prices) < period:
            return 0
        alpha = 2 / (period + 1)
        ema = prices[0]  # Start with the first price
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    ema = compute_ema(closing_prices)

    # Feature 2: Average True Range (ATR) for volatility
    def compute_atr(prices, high_prices, low_prices, period=14):
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                   abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.nan
        return atr

    high_prices = s[2::6]
    low_prices = s[3::6]
    atr = compute_atr(closing_prices, high_prices, low_prices)

    # Feature 3: 14-day RSI for momentum
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0

        if avg_loss == 0:
            return 100  # Avoid division by zero
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    features = [ema, atr, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= np.random.uniform(70, 100)  # Strong negative reward for BUY signals
        reward += np.random.uniform(10, 30)   # Mild positive reward for SELL signals
    elif risk_level > 0.4:
        reward -= np.random.uniform(30, 50)    # Moderate negative reward for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += np.random.uniform(20, 40)  # Positive reward for bullish momentum
        else:
            reward += np.random.uniform(20, 40)  # Positive reward for bearish momentum

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(15, 25)   # Reward for mean-reversion

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within the [-100, 100] range
    return np.clip(reward, -100, 100)