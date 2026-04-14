import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (0, 6, 12, ..., 114)
    volumes = s[4::6]          # Extract volumes (4, 10, 16, ..., 114)

    # Feature 1: Price Change (%)
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) > 1 else 0

    # Feature 2: Average Volume (last 20 days)
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0

    # Feature 3: Price Momentum (current close - close from 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 4: Relative Strength Index (RSI) over the last 14 days
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices)

    # Feature 5: Average True Range (ATR) for volatility estimation
    atr = np.mean(np.abs(np.diff(closing_prices[-14:]))) if len(closing_prices) >= 14 else 0

    features = [price_change, average_volume, price_momentum, rsi, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate historical thresholds based on the features
    price_change_threshold = np.std(features[0]) * 1.5
    rsi_threshold_low = 30
    rsi_threshold_high = 70

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > 0.7:
        reward += -40 if features[0] > price_change_threshold else 5  # Strong negative for buying in high risk
    elif risk_level > 0.4:
        reward += -20 if features[0] > price_change_threshold else 0  # Moderate negative for buying in elevated risk

    # **Priority 2 — TREND FOLLOWING**
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[2] > 0:  # Upward momentum
            reward += 15  # Positive reward for bullish momentum
        elif trend_direction < -0.3 and features[2] < 0:  # Downward momentum
            reward += 15  # Positive reward for bearish momentum

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < rsi_threshold_low:  # Oversold condition
            reward += 10  # Reward for buying in mean-reversion condition
        elif features[3] > rsi_threshold_high:  # Overbought condition
            reward += 10  # Reward for selling in mean-reversion condition

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds