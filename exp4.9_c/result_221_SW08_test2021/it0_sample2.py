import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices
    opening_prices = s[1::6]  # Opening prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)

        average_gain = np.mean(gain[-period:]) if len(gain) >= period else np.nan
        average_loss = np.mean(loss[-period:]) if len(loss) >= period else np.nan

        if average_loss == 0:
            return 100  # Avoid division by zero
        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = compute_rsi(closing_prices)

    # Feature 2: 5-day Moving Average
    moving_average = np.mean(closing_prices[-5:])

    # Feature 3: Volume Change
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.nan
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    features = [rsi, moving_average, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50) # for example, use random for variability
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 10
    
    # If risk is low, proceed to evaluate trend following
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:
                reward += 10  # Positive reward for upward features
            else:
                reward += 10  # Positive reward for downward features
        
        # Priority 3 — SIDEWAYS / MEAN REVERSION
        if abs(trend_direction) < 0.3:
            # Example features would indicate overbought/oversold conditions
            # Assuming we interpret features[0] (RSI) for that
            if enhanced_state[123] < 30:  # Assuming RSI < 30 is oversold
                reward += 10  # Buy signal
            elif enhanced_state[123] > 70:  # Assuming RSI > 70 is overbought
                reward += 10  # Sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)