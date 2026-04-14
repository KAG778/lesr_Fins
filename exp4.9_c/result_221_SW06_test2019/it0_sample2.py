import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: 14-day simple moving average (SMA)
    sma_period = 14
    sma = np.convolve(closing_prices, np.ones(sma_period)/sma_period, mode='valid')[-1]  # Last SMA value

    # Feature 2: 14-day Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 3: Price momentum (current close price minus the price 5 days ago)
    momentum_period = 5
    momentum = closing_prices[-1] - closing_prices[-momentum_period] if len(closing_prices) > momentum_period else 0

    features = [sma, rsi, momentum]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[1] < 30:  # Assuming RSI < 30 indicates oversold
            reward += -50  # Strong penalty for buying in high risk
        if features[1] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 10  # Mild positive reward for selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[1] < 30:  # Assuming RSI < 30 indicates oversold
            reward += -20  # Moderate penalty for buying in elevated risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[1] < 70:  # Buy signal in uptrend
            reward += 20  # Positive reward for correct bullish bet
        elif trend_direction < 0 and features[1] > 30:  # Sell signal in downtrend
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold condition
            reward += 10  # Reward for buying oversold
        elif features[1] > 70:  # Overbought condition
            reward += -10  # Penalty for buying overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce magnitude of reward by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds