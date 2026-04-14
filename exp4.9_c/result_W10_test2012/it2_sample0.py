import numpy as np

def revise_state(s):
    features = []
    
    # Extract necessary price information
    closing_prices = s[0::6]
    high_prices = s[2::6]
    low_prices = s[3::6]
    volumes = s[4::6]

    # Feature 1: Rate of change (ROC) over the last 14 days for momentum
    roc_period = 14
    roc = (closing_prices[-1] - closing_prices[-roc_period-1]) / closing_prices[-roc_period-1] if len(closing_prices) > roc_period else 0
    features.append(roc)

    # Feature 2: Average True Range (ATR) for volatility measurement
    def compute_atr():
        true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                                 np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                            np.abs(low_prices[1:] - closing_prices[:-1])))
        return np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.mean(true_ranges)  # ATR over the last 14 days
    
    atr = compute_atr()
    features.append(atr)

    # Feature 3: Moving Average Convergence Divergence (MACD)
    def compute_macd():
        short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
        long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
        macd = short_ema - long_ema
        return macd
    
    macd = compute_macd()
    features.append(macd)

    # Feature 4: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
    losses = -np.where(np.diff(closing_prices) < 0, np.diff(closing_prices), 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_risk_level = np.mean(enhanced_s[123])  # Use the average of features as a proxy for historical risk
    historical_std_risk = np.std(enhanced_s[123])
    high_risk_threshold = historical_risk_level + 1.5 * historical_std_risk
    low_risk_threshold = historical_risk_level - 1.5 * historical_std_risk

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -50  # Strong negative for risky BUY-aligned features
        return reward  # Immediate return to prioritize risk management
    elif risk_level > low_risk_threshold:
        reward += 20 if enhanced_s[123][1] < 0 else -50  # Mild positive for SELL, strong negative for BUY

    # Extract features
    features = enhanced_s[123:]

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0 and features[0] > 0:  # Upward momentum
            reward += 30  # Strong positive reward for upward alignment
        elif trend_direction < 0 and features[0] < 0:  # Downward momentum
            reward += 30  # Strong positive reward for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        if features[3] < 30:  # Oversold condition based on RSI
            reward += 30  # Reward for buying in an oversold market
        elif features[3] > 70:  # Overbought condition based on RSI
            reward += -30  # Penalty for buying in an overbought market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward