import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    days = len(closing_prices)

    # Feature 1: Price Momentum (5 days difference)
    price_momentum = np.zeros_like(closing_prices)
    for i in range(5, days):
        price_momentum[i] = closing_prices[i] - closing_prices[i - 5]
    
    # Feature 2: Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = np.zeros(days)
    for i in range(days):
        rsi[i] = compute_rsi(closing_prices[max(0, i-14):i+1])  # Calculate RSI for the last 14 days

    # Feature 3: MACD (12-26 day EMA difference)
    def compute_macd(prices, short_window=12, long_window=26):
        ema_short = np.zeros_like(prices)
        ema_long = np.zeros_like(prices)
        ema_short[:short_window] = np.mean(prices[:short_window])
        ema_long[:long_window] = np.mean(prices[:long_window])
        
        for i in range(short_window, len(prices)):
            ema_short[i] = (prices[i] * (2 / (short_window + 1))) + (ema_short[i - 1] * (1 - (2 / (short_window + 1))))
        
        for i in range(long_window, len(prices)):
            ema_long[i] = (prices[i] * (2 / (long_window + 1))) + (ema_long[i - 1] * (1 - (2 / (long_window + 1))))
        
        return ema_short - ema_long

    macd = compute_macd(closing_prices)

    # Avoiding NaN values in features due to initial calculations
    return np.array([
        price_momentum[-1],  # latest price momentum
        rsi[-1],             # latest RSI
        macd[-1]             # latest MACD value
    ])

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward = -np.random.uniform(5, 15)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        else:
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)