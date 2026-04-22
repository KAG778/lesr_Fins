import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Change Percentage from the last day
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Volume Change Percentage from the last day
    volume_change_pct = (trading_volumes[-1] - trading_volumes[-2]) / trading_volumes[-2] if trading_volumes[-2] != 0 else 0.0
    
    # Feature 3: Historical Volatility (standard deviation of the last 20 closing prices)
    historical_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    
    # Feature 4: Modified Relative Strength Index (RSI) for shorter periods
    def modified_rsi(prices, period=7):
        if len(prices) < period:
            return 50.0  # Neutral RSI if not enough data
        deltas = np.diff(prices[-period:])
        gain = np.mean(np.where(deltas > 0, deltas, 0))
        loss = -np.mean(np.where(deltas < 0, deltas, 0))
        rs = gain / loss if loss != 0 else 0
        return 100 - (100 / (1 + rs))
    
    rsi_short = modified_rsi(closing_prices)

    # Feature 5: Z-score of Recent Price Change Percentage
    mean_price_change = np.mean(np.diff(closing_prices[-10:]))
    std_price_change = np.std(np.diff(closing_prices[-10:])) if np.std(np.diff(closing_prices[-10:])) != 0 else 1
    z_score_price_change = (price_change_pct - mean_price_change) / std_price_change

    features = [price_change_pct, volume_change_pct, historical_volatility, rsi_short, z_score_price_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate relative thresholds based on historical data from features
    avg_risk = np.mean(features)  # Mean of price momentum as a proxy for risk
    std_risk = np.std(features)    # Standard deviation for risk thresholding
    risk_threshold_high = avg_risk + 2 * std_risk
    risk_threshold_mid = avg_risk + std_risk

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50) if features[0] > 0 else 0  # Strong negative for BUY signals
        reward += np.random.uniform(5, 10) if features[0] < 0 else 0   # Mild positive for SELL signals
    elif risk_level > risk_threshold_mid:
        reward -= np.random.uniform(10, 20) if features[0] > 0 else 0  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_mid:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20) if features[0] > 0 else 0  # Reward if price momentum aligns
        elif trend_direction < 0:  # Downtrend
            reward += np.random.uniform(10, 20) if features[0] < 0 else 0  # Reward if price momentum aligns

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Strongly oversold condition
            reward += np.random.uniform(10, 20)  # Reward for mean-reversion buy
        elif features[0] > 0.05:  # Strongly overbought condition
            reward += np.random.uniform(10, 20)  # Reward for mean-reversion sell

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]