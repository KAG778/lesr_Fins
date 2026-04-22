import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes
    
    # Feature 1: Historical Volatility (standard deviation of the last 20 days)
    if len(closing_prices) >= 20:
        historical_volatility = np.std(closing_prices[-20:])
    else:
        historical_volatility = 0.0
    
    # Feature 2: Modified Relative Strength Index (RSI) for shorter periods
    def modified_rsi(prices, period=7):
        if len(prices) < period:
            return 50.0  # Neutral RSI if not enough data
        deltas = np.diff(prices[-period:])
        gain = np.mean(np.where(deltas > 0, deltas, 0))
        loss = -np.mean(np.where(deltas < 0, deltas, 0))
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_short = modified_rsi(closing_prices)
    
    # Feature 3: Price Change Percentage from the last day
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 4: Volume Change Percentage from the last day
    volume_change_pct = (trading_volumes[-1] - trading_volumes[-2]) / trading_volumes[-2] if trading_volumes[-2] != 0 else 0.0
    
    features = [historical_volatility, rsi_short, price_change_pct, volume_change_pct]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    risk_threshold = np.std(features[0])  # Using historical volatility as a proxy
    trend_threshold = 0.3  # A predefined threshold for trend direction (could also be dynamic)
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:  # High risk condition
        if features[2] > 0:  # Positive price change indicates a BUY signal
            reward -= 50  # Strong negative reward
        else:  # Negative price change or hold indicates a SELL signal
            reward += 10  # Mild positive reward

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold and features[2] > 0:  # Uptrend with positive price momentum
            reward += 20  # Reward for alignment with upward trend
        elif trend_direction < -trend_threshold and features[2] < 0:  # Downtrend with negative price momentum
            reward += 20  # Reward for alignment with downward trend

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[2] < -0.05:  # Strongly oversold condition
            reward += 15  # Reward for buying in mean reversion
        elif features[2] > 0.05:  # Strongly overbought condition
            reward += 15  # Reward for selling in mean reversion

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]