import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]          # Extracting trading volumes

    # Feature 1: Price Change Percentage over the last 5 days
    price_change_pct = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Feature 2: Average True Range (ATR) for volatility over the last 14 days
    def calculate_atr(prices, high_prices, low_prices, period=14):
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - prices[:-1]), 
                                   np.abs(low_prices[1:] - prices[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    high_prices = s[2::6]
    low_prices = s[3::6]
    atr = calculate_atr(closing_prices, high_prices, low_prices)

    # Feature 3: Adaptive Relative Strength Index (RSI) with dynamic period based on volatility
    def calculate_dynamic_rsi(prices, period):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    dynamic_rsi_period = max(14, int(atr * 100))  # Dynamic period based on ATR
    rsi = calculate_dynamic_rsi(closing_prices, dynamic_rsi_period)

    # Feature 4: Price Momentum (percentage change over the last 3 days)
    price_momentum = (closing_prices[-1] - closing_prices[-4]) / closing_prices[-4] if closing_prices[-4] != 0 else 0

    features = [price_change_pct, atr, rsi, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical metrics to set dynamic thresholds
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
        reward += 20 * np.sign(trend_direction)  # Reward for momentum alignment

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within [-100, 100]
    return np.clip(reward, -100, 100)