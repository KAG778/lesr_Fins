import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    # 1. Average True Range (ATR) for volatility measure
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-20:]) if len(true_ranges) >= 20 else 0

    # 2. Price Momentum (current closing price - moving average of last 5 days)
    moving_average_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    momentum = closing_prices[-1] - moving_average_5

    # 3. Exponential Moving Average (EMA) for trend detection
    ema_short = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    ema_long = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    macd = ema_short - ema_long

    # 4. Volume Spike (current volume / average volume of last 5 days)
    avg_volume_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
    volume_spike = volumes[-1] / avg_volume_5 if avg_volume_5 > 0 else 0

    # 5. Relative Strength Index (RSI) to identify overbought/oversold conditions
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return 50  # Neutral RSI if not enough data
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = -np.where(deltas < 0, deltas, 0).mean()
        
        if (gain + loss) == 0:
            return 50  # Neutral RSI
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices[-14:])  # Using the last 14 prices for RSI

    features = [momentum, atr, macd, volume_spike, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical volatility threshold based on past 20 days
    historical_returns = enhanced_s[0:120][0::6]  # Closing prices
    if len(historical_returns) > 1:
        daily_returns = np.diff(historical_returns) / historical_returns[:-1]
        historical_volatility = np.std(daily_returns)
    else:
        historical_volatility = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # MILD POSITIVE reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        elif trend_direction < 0:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)   # Reward for mean-reversion features
        reward -= np.random.uniform(5, 10)    # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds