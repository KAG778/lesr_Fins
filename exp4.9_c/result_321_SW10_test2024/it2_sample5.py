import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    highs = s[2:120:6]            # Extract high prices
    lows = s[3:120:6]             # Extract low prices
    
    # Feature 1: Price Momentum (current closing price vs opening price)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / (closing_prices[-2] + 1e-10)

    # Feature 2: Average True Range (ATR) over the last 20 days
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closing_prices[:-1]), 
                               np.abs(lows[1:] - closing_prices[:-1])))
    atr = np.mean(tr) if len(tr) > 0 else 0.0

    # Feature 3: Volume Change (% Change from previous volume)
    volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1e-10) if len(volumes) > 1 else 0.0

    # Feature 4: Price Range normalized by closing price
    price_range = (highs[-1] - lows[-1]) / (closing_prices[-1] + 1e-10)

    # Feature 5: Historical Volatility (Standard Deviation of Returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(returns) if len(returns) > 0 else 0.0

    return np.array([price_momentum, atr, volume_change, price_range, historical_volatility])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]
    
    reward = 0.0
    
    price_momentum = features[0]
    atr = features[1]
    volume_change = features[2]
    price_range = features[3]
    historical_volatility = features[4]

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50.0 * (price_momentum if price_momentum > 0 else 1)  # Strong negative for BUY-aligned features
        reward += 10.0 * (1 - (volume_change + 1e-10))  # Mild positive for SELL-aligned features (less volume)
    elif risk_level > 0.4:
        reward -= 20.0 * price_momentum  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 15.0 * trend_direction * price_momentum  # Reward for following trend based on price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if price_momentum < -0.01:  # Oversold condition
            reward += 10.0  # Encourage BUY
        elif price_momentum > 0.01:  # Overbought condition
            reward += 10.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Final clipping to ensure reward is in the range [-100, 100]
    return float(np.clip(reward, -100, 100))