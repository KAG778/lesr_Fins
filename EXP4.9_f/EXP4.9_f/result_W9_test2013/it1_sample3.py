import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes
    features = []
    
    # Feature 1: Price Change (current price - previous price)
    price_changes = np.diff(closing_prices, prepend=closing_prices[0])
    features.append(np.mean(price_changes))  # Average price change
    
    # Feature 2: Historical Volatility (standard deviation of returns)
    daily_returns = price_changes / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(daily_returns)  # Historical volatility
    features.append(historical_volatility)
    
    # Feature 3: 5-day Moving Average of Closing Prices
    moving_average = np.convolve(closing_prices, np.ones(5) / 5, mode='valid')
    features.append(moving_average[-1])  # Latest moving average
    
    # Feature 4: Mean Reversion Indicator (Z-score of the last price)
    mean_price = np.mean(closing_prices[-5:])  # Mean of last 5 closing prices
    z_score = (closing_prices[-1] - mean_price) / np.std(closing_prices[-5:]) if np.std(closing_prices[-5:]) != 0 else 0
    features.append(z_score)
    
    # Feature 5: Price Momentum (current closing price vs. closing price 5 days ago)
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0

    # Calculate dynamic thresholds based on historical volatility
    mean_volatility = np.mean(features[1])  # Historical volatility
    dynamic_risk_threshold = 0.7 * mean_volatility  # Relative threshold for risk

    # Priority 1: RISK MANAGEMENT
    if risk_level > dynamic_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assume positive price change indicates bullish sentiment
            return -40.0  # Strong penalty for buying in dangerous conditions
        else:  # Selling aligned features
            return 10.0  # Mild positive reward for selling
    
    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += max(10 * features[4], 10)  # Positive reward for momentum in uptrend
        else:  # Downtrend
            reward += max(10 * -features[4], 10)  # Positive reward for momentum in downtrend

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        if features[3] < -1:  # Assuming negative z-score indicates oversold
            reward += 15.0  # Positive reward for buying in oversold condition
        elif features[3] > 1:  # Assuming positive z-score indicates overbought
            reward -= 15.0  # Negative reward for selling in overbought condition

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 * mean_volatility and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range