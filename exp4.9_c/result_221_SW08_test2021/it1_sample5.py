import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Price Momentum (last 3-day change)
    price_momentum = (closing_prices[-1] - closing_prices[-4]) / closing_prices[-4] if closing_prices[-4] != 0 else 0
    
    # Feature 2: Average True Range (ATR) over the last 14 days
    def calculate_atr(prices, period=14):
        highs = prices[2::6]
        lows = prices[3::6]
        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closing_prices[:-1]), 
                                   np.abs(lows[1:] - closing_prices[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0
        
    atr = calculate_atr(s)  # Calculate ATR using raw state
    
    # Feature 3: Exponential Moving Average (EMA) of the closing prices (last 10 days)
    def calculate_ema(prices, period=10):
        ema = np.zeros_like(prices)
        ema[0] = prices[0]  # Starting point
        alpha = 2 / (period + 1)
        for i in range(1, len(prices)):
            ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
        return ema[-1]  # Return the last EMA value

    ema = calculate_ema(closing_prices[-10:])  # Last 10 days EMA
    
    # Feature 4: Bollinger Bands (20-day SMA and bands)
    def calculate_bollinger_bands(prices, period=20, num_std=2):
        if len(prices) < period:
            return np.nan, np.nan, np.nan  # Not enough data
        sma = np.mean(prices[-period:])
        std_dev = np.std(prices[-period:])
        upper_band = sma + (num_std * std_dev)
        lower_band = sma - (num_std * std_dev)
        return sma, upper_band, lower_band

    sma, upper_band, lower_band = calculate_bollinger_bands(closing_prices)  # Last 20 days
    
    # Combine features into an array
    features = [price_momentum, atr, ema, upper_band, lower_band]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Relative thresholds based on historical standard deviation
    mean_risk = np.mean(risk_level)  # Example: historical mean
    std_risk = np.std(risk_level)    # Example: historical std deviation

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > (mean_risk + 1.5 * std_risk):
        reward -= 50  # Strong negative for BUY
        reward += 10   # Mild positive for SELL
    elif risk_level > (mean_risk + 0.5 * std_risk):
        reward -= 20  # Moderate negative for BUY

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < (mean_risk - 0.5 * std_risk):
        if trend_direction > 0:
            reward += 20  # Reward for bullish trend
        else:
            reward += 20  # Reward for bearish trend

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < (mean_risk - 1.0 * std_risk):
        reward += 10  # Reward mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < (mean_risk - 0.5 * std_risk):
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within [-100, 100]
    return np.clip(reward, -100, 100)