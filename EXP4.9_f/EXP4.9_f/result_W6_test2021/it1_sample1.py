import numpy as np

def revise_state(s):
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Price Momentum (last day closing price - average of last N days closing prices)
    N_momentum = 5  # Lookback period for momentum
    price_momentum = (closing_prices[-1] - np.mean(closing_prices[-N_momentum:])) / np.std(closing_prices[-N_momentum:]) if len(closing_prices) >= N_momentum else 0
    
    # Feature 2: Volatility (rolling standard deviation over the last N days)
    N_volatility = 20  # Lookback period for volatility
    volatility = np.std(closing_prices[-N_volatility:]) if len(closing_prices) >= N_volatility else 0
    
    # Feature 3: Z-score of RSI (Relative Strength Index) over the last 14 days
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        gains = np.where(np.diff(prices) > 0, np.diff(prices), 0)
        losses = np.where(np.diff(prices) < 0, -np.diff(prices), 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices[-14:])
    rsi_zscore = (rsi - 50) / 10  # Normalize RSI around 50 with a standard deviation of 10
    
    # Feature 4: Volume Spike (percentage change in volume compared to the average volume over the last N days)
    N_volume = 20  # Lookback period for volume
    avg_volume = np.mean(volumes[-N_volume:]) if len(volumes) >= N_volume else 0
    volume_spike = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    
    # Return the features as a numpy array
    features = [price_momentum, volatility, rsi_zscore, volume_spike]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive momentum indicates a buy signal
            reward = -50
        # Mild positive reward for SELL-aligned features
        elif features[0] < 0:  # Negative momentum indicates a sell signal
            reward = 10
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward = -20
    
    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and positive momentum
            reward += 20
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and negative momentum
            reward += 20
    
    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] > 70:  # Overbought condition, sell signal
            reward += 15
        elif features[2] < 30:  # Oversold condition, buy signal
            reward += 15

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified bounds