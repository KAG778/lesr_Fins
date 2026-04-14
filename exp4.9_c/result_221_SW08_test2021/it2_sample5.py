import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]          # Extracting trading volumes

    # Feature 1: Price Change Percentage over the last 5 days
    if len(closing_prices) >= 6:
        price_change_pct = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
    else:
        price_change_pct = 0

    # Feature 2: Average True Range (ATR) over the last 14 days
    def calculate_atr(prices):
        highs = prices[2::6]
        lows = prices[3::6]
        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closing_prices[:-1]), 
                                   np.abs(lows[1:] - closing_prices[:-1])))
        return np.mean(tr[-14:]) if len(tr) >= 14 else 0

    atr = calculate_atr(s)  # Calculate ATR using raw state

    # Feature 3: Dynamic Relative Strength Index (RSI) based on volatility
    def compute_dynamic_rsi(prices, period):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        average_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        average_loss = np.mean(loss[-period:] if len(loss) >= period else 0)
        rs = average_gain / average_loss if average_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    volatility = np.std(np.diff(closing_prices) / closing_prices[:-1]) if len(closing_prices) > 1 else 0
    dynamic_rsi_period = max(14, int(volatility * 100))  # Dynamic period based on volatility
    rsi = compute_dynamic_rsi(closing_prices, dynamic_rsi_period)

    # Feature 4: Price Momentum (percentage change over the last 3 days)
    price_momentum = (closing_prices[-1] - closing_prices[-4]) / closing_prices[-4] if len(closing_prices) >= 4 else 0

    features = [price_change_pct, atr, rsi, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical data
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
        if trend_direction > trend_threshold:  # Uptrend
            reward += 20  # Reward for bullish momentum
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 20  # Reward for bearish momentum

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within [-100, 100]
    return np.clip(reward, -100, 100)