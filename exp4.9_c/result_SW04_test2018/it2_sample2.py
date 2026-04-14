import numpy as np

def revise_state(s):
    features = []
    
    # Extracting the closing prices for the last 20 days
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]      # High prices
    low_prices = s[3::6]       # Low prices
    volumes = s[4::6]          # Trading volumes
    
    # Feature 1: Average True Range (ATR) - captures volatility
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR
    features.append(atr)

    # Feature 2: Price Momentum (current closing price vs previous closing price)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(momentum)

    # Feature 3: Volume Change - compares current volume vs average volume over last 5 days
    avg_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 1  # Avoid division by zero
    volume_change = (volumes[-1] - avg_volume) / avg_volume  # Normalized volume change
    features.append(volume_change)

    # Feature 4: Relative Strength Index (RSI) for the last 14 days
    window_length = 14
    rsi = 50  # Default to neutral
    if len(closing_prices) > window_length:
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)

        avg_gain = np.mean(gains[-window_length:])
        avg_loss = np.mean(losses[-window_length:])
        rs = avg_gain / avg_loss if avg_loss > 0 else np.inf
        rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(enhanced_s[123:127])  # Use features for volatility context
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.4 * historical_volatility
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong penalty for risky buy signals
        reward += np.random.uniform(5, 10)    # Mild positive for sell signals
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(5, 15)  # Moderate penalty for buy signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 20 * (1 if trend_direction > 0 else -1)  # Positive reward for trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        # Reward mean-reversion features (assumed features are in enhanced_s[123:])
        reward += 10  # Example positive reward for mean-reversion alignment

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_volatility and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Constrain the reward within [-100, 100]
    return np.clip(reward, -100, 100)