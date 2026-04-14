import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]          # Extracting trading volumes

    # Feature 1: Rate of Change (ROC) over the last 10 days
    roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if closing_prices[-11] != 0 else 0

    # Feature 2: Exponential Moving Average (EMA) of the last 10 days
    def calculate_ema(prices, period=10):
        ema = np.zeros_like(prices)
        alpha = 2 / (period + 1)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
        return ema[-1]

    ema = calculate_ema(closing_prices[-10:])  # Last 10 days EMA

    # Feature 3: Volume Change Percentage from the average of the last 20 days
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    volume_change_pct = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    # Feature 4: 14-day Average True Range (ATR) for volatility
    def calculate_atr(prices):
        highs = prices[2::6]
        lows = prices[3::6]
        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closing_prices[:-1]), 
                                   np.abs(lows[1:] - closing_prices[:-1])))
        return np.mean(tr)

    atr = calculate_atr(s)  # Calculate ATR using raw state

    features = [roc, ema, volume_change_pct, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical std deviation of the features
    historical_std = np.std(enhanced_s[123:])  # Assuming features start from index 123
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Reward for following the trend

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is bounded within [-100, 100]
    return np.clip(reward, -100, 100)