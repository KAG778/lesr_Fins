import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Closing prices at every 6th index
    volumes = s[4::6]          # Trading volumes at every 6th index starting from index 4
    num_days = len(closing_prices)

    # Feature 1: 14-day Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return np.nan
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0).mean()
        loss = np.abs(np.where(delta < 0, delta, 0)).mean()
        rs = gain / loss if loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    # Feature 2: 14-day Average True Range (ATR) for volatility
    def compute_atr(prices, high, low, period=14):
        tr = np.maximum(high[1:] - low[1:], high[1:] - closing_prices[:-1], closing_prices[:-1] - low[1:])
        return np.mean(tr[-period:]) if len(tr) >= period else np.nan

    if num_days >= 15:
        highs = s[1::6]  # High prices
        lows = s[2::6]   # Low prices
        atr = compute_atr(closing_prices, highs, lows)
    else:
        atr = np.nan

    # Feature 3: Price Momentum (current vs previous day)
    if num_days > 1:
        momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        momentum = 0

    # Feature 4: Volume Change (current vs previous day)
    if num_days > 1:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
    else:
        volume_change = 0

    features = [
        rsi if not np.isnan(rsi) else 50,  # Neutral if not enough data
        atr if not np.isnan(atr) else 0,    # Handle NaN for ATR
        momentum,                            # Price momentum
        volume_change                        # Volume change
    ]

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Compute historical thresholds for risk management
    historical_volatility = np.std(features[1]) if features[1] != 0 else 1  # Prevent division by zero
    risk_threshold = 0.7 * historical_volatility  # Relative threshold based on historical std
    trend_threshold = 0.3  # Fixed trend threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 40  # STRONG NEGATIVE for BUY-aligned features
        reward += 10 * features[1]  # Mild positive for SELL-aligned features based on ATR

    elif risk_level > (0.4 * historical_volatility):
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < (0.4 * historical_volatility):
        if trend_direction > trend_threshold:  # Uptrend
            reward += 20 * features[2]  # Reward based on momentum
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 20 * -features[2]  # Reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < (0.3 * historical_volatility):
        # Reward mean-reversion features
        if features[0] < 30:  # RSI indicating oversold
            reward += 10  # Buy signal
        elif features[0] > 70:  # RSI indicating overbought
            reward += 10  # Sell signal
        else:
            reward -= 10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < (0.4 * historical_volatility):
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within bounds