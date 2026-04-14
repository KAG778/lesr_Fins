import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    days = len(closing_prices)

    # Feature 1: 14-day Relative Strength Index (RSI) - to gauge momentum
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    # Feature 2: Bollinger Bands Width (20-day)
    if days >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        bb_width = (rolling_std * 2) / rolling_mean if rolling_mean != 0 else 0
    else:
        bb_width = 0

    # Feature 3: Average True Range (ATR) - to measure volatility
    true_ranges = np.maximum(
        closing_prices[1:] - closing_prices[:-1], 
        np.maximum(
            np.abs(closing_prices[1:] - closing_prices[:-1]), 
            np.abs(closing_prices[1:] - closing_prices[:-1])
        )
    )
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR

    # Feature 4: Price Momentum (recent 5-day momentum)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if days > 5 and closing_prices[-6] != 0 else 0

    # Collect features
    features = [rsi, bb_width, atr, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for relative thresholds
    # These would typically be derived from historical data; using placeholders for illustration
    historical_mean_risk = 0.5  # Placeholder: Replace with actual historical mean risk
    historical_std_risk = 0.2     # Placeholder: Replace with actual historical std of risk

    # Define relative thresholds for risk
    high_risk_threshold = historical_mean_risk + historical_std_risk
    low_risk_threshold = historical_mean_risk - historical_std_risk

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
        reward += 20   # Mild positive for SELL
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 30 * np.sign(trend_direction)  # Strong reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)