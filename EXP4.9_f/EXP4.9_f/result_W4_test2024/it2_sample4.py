import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices for 20 days
    volumes = s[4:120:6]          # Trading volumes for 20 days
    days = len(closing_prices)

    # Feature 1: Z-Score of Closing Prices
    mean_price = np.mean(closing_prices) if days > 0 else 0
    std_price = np.std(closing_prices) if days > 0 else 1e-10  # Avoid division by zero
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0

    # Feature 2: Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if days >= 5 and closing_prices[-5] != 0 else 0

    # Feature 3: Volume Change Percentage
    volume_change_pct = (volumes[-1] - volumes[-2]) / volumes[-2] if days >= 2 and volumes[-2] != 0 else 0

    # Feature 4: Relative Strength Index (RSI) for market momentum
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0  # Last 14 days
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0  # Last 14 days

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    features = [z_score, roc, volume_change_pct, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Use features to calculate a standard deviation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # STRONG NEGATIVE for BUY-aligned features
        # Mild positive for SELL-aligned features
        reward += 10 if trend_direction < 0 else 0
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Positive reward for trend-following

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]