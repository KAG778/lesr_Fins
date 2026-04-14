import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Price Momentum (last closing price - average of last 5 closing prices)
    momentum = closing_prices[-1] - np.mean(closing_prices[-6:-1]) if len(closing_prices) >= 6 else 0.0

    # Feature 2: Average Daily Volume (last 5 trading days)
    avg_volume = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else 0.0

    # Feature 3: Volatility (using standard deviation of last 5 days of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0.0

    # Feature 4: Relative Strength Index (RSI) - 14 days
    if len(closing_prices) < 14:
        rsi = 0.0  # Not enough data to calculate RSI
    else:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = -np.where(deltas < 0, deltas, 0).mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

    # Feature 5: Price Change Percentage (last day vs. the day before)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 and closing_prices[-2] != 0 else 0.0

    # Return only new features
    return np.array([momentum, avg_volume, volatility, rsi, price_change_pct])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate historical thresholds (mean and std) for dynamic thresholds
    historical_risk_threshold = np.mean([0.4, 0.5, 0.6])  # Example historical values for risk level in a real environment
    historical_trend_threshold = 0.3
    historical_volatility_threshold = 0.6

    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_risk_threshold:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10   # Mild positive reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > historical_trend_threshold and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Positive reward for upward momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < historical_trend_threshold and risk_level < 0.3:
        rsi = enhanced_s[123]  # Assuming RSI is included in revised state
        if rsi < 30:  # Oversold condition
            reward += 20  # Strong reward for potential buy
        elif rsi > 70:  # Overbought condition
            reward += 20  # Strong reward for potential sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return float(np.clip(reward, -100, 100))