import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices for 20 days
    high_prices = s[2:120:6]     # High prices for 20 days
    low_prices = s[3:120:6]      # Low prices for 20 days
    volumes = s[4:120:6]         # Trading volumes for 20 days

    days = len(closing_prices)

    # Feature 1: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:]) if days >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if days >= 26 else 0
    macd = short_ema - long_ema

    # Feature 2: Rate of Change (momentum indicator)
    if days > 5:
        roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    else:
        roc = 0

    # Feature 3: Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices - low_prices, 
                    np.maximum(np.abs(high_prices - np.roll(closing_prices, 1)), 
                               np.abs(low_prices - np.roll(closing_prices, 1))))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 4: Price Relative to 200-day Moving Average
    if days >= 200:
        ma_200 = np.mean(closing_prices[-200:])
        price_relative_to_ma200 = (closing_prices[-1] - ma_200) / ma_200 if ma_200 != 0 else 0
    else:
        price_relative_to_ma200 = 0

    features = [macd, roc, atr, price_relative_to_ma200]
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
        reward += 10 if trend_direction < 0 else 0  # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Positive reward for trend-following

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]