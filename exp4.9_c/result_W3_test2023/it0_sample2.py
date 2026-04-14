import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    highs = s[2::6]            # Extract high prices
    lows = s[3::6]             # Extract low prices
    volumes = s[4::6]          # Extract trading volumes
    adjusted_closes = s[5::6]  # Extract adjusted closing prices

    # Feature 1: Daily Returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    # Handle edge cases
    daily_returns = np.concatenate(([0], daily_returns))  # Padding to maintain length

    # Feature 2: Moving Average (10-day)
    moving_avg = np.convolve(closing_prices, np.ones(10)/10, mode='valid')
    moving_avg = np.concatenate(([np.nan]*9, moving_avg))  # Padding to maintain length

    # Feature 3: Relative Strength Index (RSI)
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)

    avg_gain = np.convolve(gains, np.ones(14)/14, mode='valid')
    avg_loss = np.convolve(losses, np.ones(14)/14, mode='valid')

    rs = avg_gain / avg_loss if avg_loss.any() else np.zeros_like(avg_gain)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate(([np.nan]*13, rsi))  # Padding to maintain length

    # Combining features into a single array
    features = np.array([
        daily_returns[-1],  # Last day's return
        moving_avg[-1] if not np.isnan(moving_avg[-1]) else 0,  # Last moving average value
        rsi[-1] if not np.isnan(rsi[-1]) else 0  # Last RSI value
    ])

    return np.array(features)

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
        if enhanced_s[123] == 0:  # Assuming BUY-aligned features
            return np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:  # Assuming SELL-aligned features
            return np.random.uniform(5, 10)  # Mild positive reward for SELL

    elif risk_level > 0.4:
        if enhanced_s[123] == 0:  # Assuming BUY-aligned features
            reward += np.random.uniform(-10, -5)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] == 0:  # Upward features
            reward += np.random.uniform(10, 20)  # Positive reward for correct upward bet
        elif trend_direction < -0.3 and enhanced_s[123] == 1:  # Downward features
            reward += np.random.uniform(10, 20)  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        if enhanced_s[123] == 0:  # Oversold feature
            reward += np.random.uniform(5, 15)  # Positive reward
        else:  # Overbought feature
            reward += np.random.uniform(-5, -15)  # Negative reward for breakout-chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]