import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices, high prices, low prices, and volumes
    closing_prices = s[0::6]
    high_prices = s[2::6]
    low_prices = s[3::6]
    volumes = s[4::6]

    # Feature 1: Recent Price Momentum (last closing price vs the one before)
    if len(closing_prices) > 1:
        momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        momentum = 0  # Default to zero if not enough data
    features.append(momentum)

    # Feature 2: 14-day Relative Strength Index (RSI) to capture overbought/oversold conditions
    window_length = 14
    rsi = 50  # Default to neutral
    if len(closing_prices) >= window_length:
        deltas = np.diff(closing_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-window_length:])
        avg_loss = np.mean(losses[-window_length:])
        rs = avg_gain / avg_loss if avg_loss > 0 else np.inf
        rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 3: Average True Range (ATR) - captures volatility to identify potential crisis
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    features.append(atr)

    # Feature 4: Volume Change Ratio (current volume vs last 5-day average volume)
    avg_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 1  # Avoid division by zero
    volume_change_ratio = (volumes[-1] - avg_volume) / avg_volume
    features.append(volume_change_ratio)

    # Feature 5: Price Range (High - Low of the last day normalized by closing price)
    if len(high_prices) > 0 and len(low_prices) > 0:
        price_range = (high_prices[-1] - low_prices[-1]) / closing_prices[-1] if closing_prices[-1] != 0 else 0
    else:
        price_range = 0
    features.append(price_range)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data for relative risk assessment
    historical_volatility = np.std(enhanced_s[123:])  # Use features from revise_state for volatility
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.4 * historical_volatility

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for risky BUY signals
        reward += np.random.uniform(5, 10)    # Mild positive for SELL signals
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(5, 15)    # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 20 * np.sign(trend_direction)  # Positive reward for alignment with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 10  # Reward for mean-reversion strategy in a sideways market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 1.5 * historical_volatility and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50% during high volatility periods

    # Constrain reward within [-100, 100]
    return np.clip(reward, -100, 100)