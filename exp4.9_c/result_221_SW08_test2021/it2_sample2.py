import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: Price Change Rate (1-day change percentage)
    price_change_rate = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: 14-day Average True Range (ATR) for volatility
    def calculate_atr(prices, period=14):
        highs = prices[2::6]
        lows = prices[3::6]
        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closing_prices[:-1]), 
                                   np.abs(lows[1:] - closing_prices[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0
        
    atr = calculate_atr(s)  # Calculate ATR using raw state

    # Feature 3: 14-day Rolling Standard Deviation of Returns (for dynamic volatility measure)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    rolling_volatility = np.std(returns[-14:]) if len(returns) >= 14 else 0

    # Feature 4: Momentum Indicator (difference between 3-day and 10-day Simple Moving Averages)
    sma_3 = np.mean(closing_prices[-3:]) if len(closing_prices) >= 3 else 0
    sma_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0
    momentum_indicator = sma_3 - sma_10

    # Feature 5: Rate of Change (ROC) for near-term price momentum
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    features = [price_change_rate, atr, rolling_volatility, momentum_indicator, roc]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(enhanced_s[123:])  # Assuming features start from index 123
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY signals
        reward += 10   # Mild positive reward for SELL signals
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > trend_threshold:  # Uptrend
            reward += 20  # Reward for bullish trend
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 20  # Reward for bearish trend

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)