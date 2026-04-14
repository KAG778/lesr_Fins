import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices of the last 20 days
    volumes = s[4::6]          # Trading volumes of the last 20 days

    # Feature 1: Price Momentum (current closing price vs previous closing price)
    momentum = closing_prices[-1] - closing_prices[-2]  # Current day - Previous day
    momentum_feature = momentum / (closing_prices[-2] if closing_prices[-2] != 0 else 1)  # Normalize

    # Feature 2: Price Range (current day high - current day low)
    high_prices = s[2::6]      # High prices of the last 20 days
    low_prices = s[3::6]       # Low prices of the last 20 days
    price_range = high_prices[-1] - low_prices[-1]
    price_range_feature = price_range / (closing_prices[-1] if closing_prices[-1] != 0 else 1)  # Normalize

    # Feature 3: 14-day RSI to capture overbought/oversold conditions
    window_length = 14
    rsi = 50  # Default to neutral
    if len(closing_prices) >= window_length:
        gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
        losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)

        avg_gain = np.mean(gains[-window_length:])
        avg_loss = np.mean(losses[-window_length:])

        rs = avg_gain / avg_loss if avg_loss > 0 else np.inf
        rsi = 100 - (100 / (1 + rs))

    # Feature 4: Volume Change (current volume vs average volume of the last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 0
    volume_change = volumes[-1] - avg_volume
    volume_change_feature = volume_change / (avg_volume if avg_volume != 0 else 1)  # Normalize

    # Return the features as a numpy array
    features = [momentum_feature, price_range_feature, rsi, volume_change_feature]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds for risk management
    historical_risk_threshold = 0.5  # This should be dynamically calculated based on historical data
    historical_trend_threshold = 0.3  # This should also be dynamically calculated

    # Initialize reward
    reward = 0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_risk_threshold:
        # Strong negative reward for BUY-aligned features
        reward -= 40  # Strong negative reward for BUY signals
        # Mild positive reward for SELL-aligned features
        reward += 10  # Mild positive reward for SELL signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > historical_trend_threshold and risk_level < historical_risk_threshold:
        # Reward based on momentum alignment
        if trend_direction > 0:
            reward += 20  # Positive reward for bullish features
        elif trend_direction < 0:
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < historical_trend_threshold and risk_level < 0.3:
        # Reward for mean-reversion
        reward += 15  # Example positive reward for mean-reversion alignment

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Constrain reward within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward