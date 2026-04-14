import numpy as np

def revise_state(s):
    # Extract closing prices and volumes from the raw state
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    volumes = s[4::6]  # Every 6th element starting from index 4
    
    # Edge case handling
    if len(closing_prices) < 2 or len(volumes) < 2:
        return np.zeros(4)  # Return zeros if there are not enough days of data

    # Feature 1: Price Momentum (recent close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2]

    # Feature 2: Price Change Percentage ((close - open) / open) for the last day
    opening_price = s[1::6][-1]  # Opening price of the last day
    price_change_percentage = ((closing_prices[-1] - opening_price) / opening_price) * 100 if opening_price != 0 else 0

    # Feature 3: Volume Change (recent volume - previous volume)
    volume_change = volumes[-1] - volumes[-2]

    # Feature 4: Average True Range (ATR) as a volatility measure
    true_ranges = [closing_prices[i] - closing_prices[i - 1] for i in range(1, len(closing_prices))]
    atr = np.mean(np.abs(true_ranges[-14:])) if len(true_ranges) >= 14 else np.mean(np.abs(true_ranges))

    features = [price_momentum, price_change_percentage, volume_change, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical std for relative thresholds
    historical_returns = enhanced_s[:120]  # Assuming this contains daily returns
    historical_std = np.std(historical_returns)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -70  # Strong negative reward for BUY signals in high-risk environments
        reward += 10    # Mild positive reward for SELL signals in high-risk environments
    elif risk_level > 0.4:
        reward += -30  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += np.clip(50 * (enhanced_s[123] / historical_std), 0, 100)  # Rewards aligned with positive momentum
        else:  # Downtrend
            reward += np.clip(50 * (enhanced_s[123] / historical_std), 0, 100)  # Rewards aligned with negative momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] > 0:  # Oversold
            reward += 30  # Positive reward for buying oversold
        elif enhanced_s[123] < 0:  # Overbought
            reward += 30  # Positive reward for selling overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]